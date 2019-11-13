
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import fsmn_pytorch
import math

from utils import to_tensor, to_numpy
from sklearn.cluster import KMeans
from asr.trainer.args import get_scp
from asr.utils import SequenceDataset
from asr.utils.common import *
from torch.multiprocessing import Process, Queue


class Env:
    def __init__(self, model, args):
        self.model = model
        self.log_writer = None
        self.ckp = args.ckp
        self.action_start = args.action_start
        self.action_end = args.action_end

        # quantify
        self.compressible_layer_types = [torch.nn.modules.linear.Linear, fsmn_pytorch.FSMN]
        self.total_size = 0     # total size of all parameters
        self.layer_size = []    # size of parameters of each layer
        self.layer_params = []  # kinds of parameters of each layer
        self._build_idx()
        self._build_state_embedding()
        self.cur_ind = 0    # current index of layer
        self.cur_size = 0   # current size of compressed layers
        self.strategy_bits = []
        self.compressed_params = {}
        self.quantify_kind = args.quantify_kind

        # validate
        _, self.cv_scp = get_scp(args, world_size=1, rank=0)
        self.valid_set = SequenceDataset(*self.cv_scp, 32, targets_delay=args.targets_delay,
                                    skip_frame=args.skip_frames, hybrid=True,
                                    feature_transform=args.feature_transform)
        self.valid_queue = Queue(50)
        self.valid_loader = Process(target=self.valid_set, args=(self.valid_queue, True), daemon=True)
        self.valid_loader.start()
        self.epoch_accumor = Metric()
        self.trajectory = 0

        # reward
        self.best_reward = -math.inf
        self.best_strategy = None

    def _build_idx(self):
        self.compressible_idx = []
        for i, layer in enumerate(self.model.modules()):
            if type(layer) in self.compressible_layer_types:
                self.compressible_idx.append(i)

                size = 0
                kind = 0
                for param in layer.parameters():
                    size += np.prod(param.data.size())
                    kind += 1

                self.total_size += size * 32
                self.layer_size.append(size)
                self.layer_params.append(kind)

        self.compressible_length = len(self.compressible_idx)

    def _build_state_embedding(self):
        # build the static part of the state embedding of each layer
        layer_embedding = []
        modules = list(self.model.modules())
        for i, idx in enumerate(self.compressible_idx):
            layer = modules[idx]
            this_state = [i]    # the layer index
            if type(layer) == nn.Linear:
                this_state.append(0)    # layer type, 0 for linear
                this_state.append(layer.in_features)    # in channels
                this_state.append(layer.out_features)   # out channels
                this_state.append(1)
                this_state.append(np.prod(layer.weight.size()))
            else:
                this_state.append(1)    # layer type, 1 for fsmn
                this_state.append(layer.filter.shape[1])    # in channels
                this_state.append(layer.filter.shape[1])    # out channels
                this_state.append(layer.filter.shape[0])    # l_order + r_order + 1
                this_state.append(np.prod(layer.filter.size()))

            this_state.append(1)    # the action of last layer
            layer_embedding.append(this_state)

        # normalize the state
        layer_embedding = np.array(layer_embedding, 'float')
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding

    def step(self, action, show=False):
        self.strategy_bits.append(action)
        self.cur_size += self.layer_size[self.cur_ind] * action + \
                         self.layer_params[self.cur_ind] * (2 ** action) * 32

        if self.cur_ind == self.compressible_length - 1:    # the final layer
            # get reward
            if not show:
                self.quantify_model()
            loss, acc = self._validate()
            compress_ratio = 1. * self.cur_size / self.total_size
            reward = -loss * math.log(self.cur_size)
            self.log_writer.write('# {}: episode_reward: {:.4f} loss: {:.4f} acc: {:.4f}, ratio: {:.4f}, best_reward: {:.4f}\n'.format(
                                    self.trajectory, reward, loss, acc, compress_ratio, self.best_reward))
            self.log_writer.flush()            

            self.trajectory += 1

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_strategy = self.strategy_bits.copy()
                self.log_writer.write('best_reward: {} best_strategy: {}\n'.format(self.best_reward, self.best_strategy))
                self.log_writer.flush()

            info_set = {'loss': loss, 'acc': acc, 'compress_ratio': compress_ratio}
            state_next = self.layer_embedding[self.cur_ind, :].copy()
            done = True
            # export ???
            return state_next, reward, done, info_set

        info_set = None
        reward = 0
        done = False
        self.cur_ind += 1  # the index of next layer
        # build next state
        self.layer_embedding[self.cur_ind][-1] = (self.strategy_bits[-1] - self.action_start) * 1. / \
                                                 (self.action_end - self.action_start)  # last action
        state_next = self.layer_embedding[self.cur_ind, :].copy()

        return state_next, reward, done, info_set

    def _forward_model(self, batch):
        if len(batch) == 2:
            feat, ali = batch
            feat = torch.tensor(feat, dtype=torch.float32).cuda().unsqueeze_(1)
            length = torch.tensor([feat.shape[0]], dtype=torch.int32)
        else:
            feat, ali, length, _ = batch
            feat = torch.tensor(feat, dtype=torch.float32).cuda()
            length = torch.tensor(length, dtype=torch.int32)
            mask = torch.arange(feat.shape[0], dtype=torch.int32).view(-1, 1) < \
                   length.view(1, -1)

        ali = torch.tensor(ali).long().view(-1).cuda()
        length = length.cuda(non_blocking=True)

        output = self.model(feat, length=length)
        max_val, prediction = torch.max(output, 2)
        acc = torch.sum(ali == prediction.view(-1)).item()
        frames = length.sum().item()

        output = output.view(-1, output.size(-1))
        loss = F.cross_entropy(output, ali, ignore_index=-1, reduction='sum')

        return {'loss': loss.item(), 'frames': frames, 'acc': acc}, loss

    def _validate(self):
        self.model.eval()
        self.epoch_accumor.clear()
        # self.log_writer.write('# {}: start validate\n'.format(self.trajectory))
        # self.log_writer.flush()
        # print('start validate')
        with torch.no_grad():
            while True:
                # self.log_writer.write('prepare batch\n')
                # self.log_writer.flush()
                # print('prepare batch\n')
                batch = self.valid_queue.get()
                # self.log_writer.write('get batch\n')
                # self.log_writer.flush()
                # print('get batch\n')
                if batch is None:
                    break
                statics, loss = self._forward_model(batch)
                del loss    # loss ???
                torch.cuda.empty_cache()
                self.epoch_accumor(statics)

        # print('end validate')
        stats = self.epoch_accumor.reduce
        loss = stats['loss'] / stats['frames']
        acc = 100 * stats['acc'] / stats['frames']
        # self.log_writer.write('# {}: batches {}, frames {}, loss {:.2f}, acc {:.2f}\n'.format(self.trajectory, self.epoch_accumor.step, stats['frames'], loss, acc))
        # self.log_writer.flush()

        return loss, acc

    def quantify_model(self):
        modules = list(self.model.named_modules())
        for i, ind in enumerate(self.compressible_idx):
            layer_name = modules[ind][0]
            layer = modules[ind][1]
            for name, param in layer.named_parameters():
                param_name = '{}.{}_{}bit'.format(layer_name, name, self.strategy_bits[i])
                # self.log_writer.write(param_name + '\n')
                # self.log_writer.flush()

                if not self.compressed_params.__contains__(param_name):  # quantify parameters
                    num = 2 ** self.strategy_bits[i]
                    # if not type(num) == int:
                        # print("num is not integer")
                        # print(type(num))

                    if self.quantify_kind == 'linear':
                        min = param.data.min()
                        unit = (param.data.max() - min) / num
                        index = (param.data - min) / unit
                        index = index.int()
                        index[index == num] -= 1
                    else:
                        tmp_param = param.data.view(-1)
                        if num >= len(tmp_param):   # do nothing
                            index = to_tensor(np.array(range(len(tmp_param)))).int()
                            index = index.view(param.data.shape)
                        else:   # k-means
                            estimator = KMeans(n_clusters=num)
                            estimator.fit(tmp_param.reshape(-1, 1).cpu())
                            index = torch.Tensor(estimator.labels_).int()
                            index = index.view(param.data.shape)

                    center = [param.data[index == i].mean().item() for i in range(num)]
                    self.compressed_params[param_name + 'c'] = to_tensor(center)
                    self.compressed_params[param_name + 'i'] = index

                idx = self.compressed_params[param_name + 'i'].long()
                param.data.copy_(self.compressed_params[param_name + 'c'][idx])

    def reset(self):
        self.model.load_state_dict(self.ckp)
        self.cur_ind = 0
        self.cur_size = 0
        self.strategy_bits = []
        self.layer_embedding[:, -1] = 1.
        state = self.layer_embedding[0].copy()

        return state

    def set_log(self, log_writer):
        self.log_writer = log_writer
