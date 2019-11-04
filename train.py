
import argparse
import numpy as np
import torch

from asr.models import ModelCreator
from rl.agent import DDPG
from rl.env import Env
from utils import *

from copy import deepcopy
from tensorboardX import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description='asr compress search')

    # env
    parser.add_argument('--ckp', default='', type=str, help='checkpoint to compress')

    # agent
    parser.add_argument('--hidden1', defalut=400, type=int)
    parser.add_argument('--hidden2', default=300, type=int)
    parser.add_argument('--rmsize', default=100, type=int, help='memory size for each layer')

    # train
    parser.add_argument('--output', default='./compress_search', type=str)
    parser.add_argument('--episode', default=50, type=int, help='episode of training')
    parser.add_argument('--seed', default=2019, type=int, help='random seed to set')
    parser.add_argument('--warmup', default=10, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size for train agent')

    return parser.parse_args()


def create_model(ckp, n_gpu=1):
    checkpoint = torch.load(ckp, map_location='cpu')
    hparams = checkpoint['hparams']
    net = ModelCreator(hparams, True)().cuda()
    net.load_state_dict(checkpoint['model'])
    if n_gpu > 1:
        net = torch.nn.DataParallel(net, range(n_gpu))

    return net


def train(num_episode, agent, env, output):
    T = []  # trajectory

    for cur_episode in range(num_episode):
        # reset as it is the start of an episode
        state = deepcopy(env.reset())

        while True:
            # agent pick action
            if cur_episode <= args.warmup:
                action = agent.random_action()
            else:
                action = agent.select_action(state)

            # env response with next_observation, reward, terminate_info
            next_state, reward, done, info = env.step(action)
            T.append([deepcopy(state), action, done])

            state = deepcopy(next_state)

            if done:    # end of episode
                episode_reward = reward
                print('#{}: episode_reward: {:.4f} acc: {:.4f}, ratio: {:.4f}'.format(
                    cur_episode, episode_reward, info['acc'], info['compress_ratio']))

                # agent observe and update policy
                for s_t, a_t, done in T:
                    agent.append_replay(s_t, a_t, episode_reward, done)
                    if cur_episode > args.warmup:
                        agent.update_policy()

                T = []

                tf_writer.add_scalar('reward/last', episode_reward, cur_episode)
                tf_writer.add_scalar('reward/best', env.best_reward, cur_episode)
                tf_writer.add_scalar('info/acc', info['acc'], cur_episode)
                tf_writer.add_scalar('info/comress_ratio', info['compress_ratio'], cur_episode)

                log_writer.write('best reward: {}\n'.format(env.best_reward))
                log_writer.write('best policy: {}\n'.format(env.best_strategy))

    log_writer.close()


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    model = create_model(args.ckp, args.n_gpu)

    env = Env(model, args)

    if args.job == 'train':
        # build folder and log
        folder_name = ''
        args.output = get_output_folder(args.output, folder_name)
        print('=> Saving logs to {}'.format(args.output))
        tf_writer = SummaryWriter(logdir=args.output)
        log_writer = open(os.path.join(args.output, 'log.txt'), 'w')
        env.set_log(log_writer)

        args.rmsize = args.rmsize * env.compressible_length
        n_state = env.layer_embedding.shape[1]
        agent = DDPG(n_state, args)

        train(args.episode, agent, env, args.output)

