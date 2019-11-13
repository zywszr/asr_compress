
import os
import random
import argparse
import numpy as np
import torch
torch.backends.cudnn.deterministic = True

from asr.utils.common import str2bool
from asr.models import ModelCreator
from rl.agent import DDPG
from rl.env import Env
from utils import get_output_folder

from copy import deepcopy
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='asr compress search')

    # env
    parser.add_argument('--ckp', default='', type=str, help='checkpoint to compress')

    # agent
    parser.add_argument('--hidden1', default=400, type=int)
    parser.add_argument('--hidden2', default=300, type=int)
    parser.add_argument('--rmsize', default=100, type=int, help='memory size for each layer')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size for train agent')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for critic')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
    parser.add_argument('--discount', default=1., type=float)
    parser.add_argument('--tau', default=0.1, type=float, help='moving average for target network')
    parser.add_argument('--moving_alpha', default=0.5, type=float)
    parser.add_argument('--action_start', default=3, type=int)
    parser.add_argument('--action_end', default=8, type=int)
    parser.add_argument('--quantify_kind', default='linear', type=str)
    parser.add_argument('--save_internal', default=50, type=int)

    # train
    parser.add_argument('--job', default='train', type=str)
    parser.add_argument('--output', default='./rl/search', type=str)
    parser.add_argument('--episode', default=1000, type=int, help='episode of training')
    parser.add_argument('--seed', default=2019, type=int, help='random seed to set')
    parser.add_argument('--warmup', default=20, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--n-gpu', default=1, type=int, help='number of gpu to use')
    parser.add_argument('--resume', default='rl/search/tau0.1-run22', type=str)
    parser.add_argument('--resumenum', default=5, type=int)

    # dataset
    parser.add_argument('--hybrid', type=str2bool, default=False)
    parser.add_argument('--train-data-dir', type=str, default='data-fbank/train_nodup_tr90',
                        help='folder of train set data')
    parser.add_argument('--cv-data-dir', type=str, default='data-fbank/train_nodup_cv10',
                        help='validation data set dir')
    parser.add_argument('--feature-transform', type=str, default='exp/final.feature_transform',
                        help='feature transform')
    parser.add_argument('--delta-order', type=int, default=0)
    parser.add_argument('--splice', type=int, nargs=2, default=[2, 2],
                        help='splice window, [left, right]')
    parser.add_argument('--cmvn-opts', type=str, default='')
    parser.add_argument('--online-cmvn', type=str2bool, default=False)
    
    # process
    parser.add_argument('--targets-delay', type=int, default=0, help='delay targets')
    parser.add_argument('--skip-frames', type=int, default=1, help='skip frames')
    
    # debug
    parser.add_argument('--show', type=str2bool, default=False, help='')

    return parser.parse_args()


def create_model(ckp, n_gpu=1):
    checkpoint = torch.load(ckp, map_location='cpu')
    hparams = checkpoint['hparams']
    net = ModelCreator(hparams, True)().cuda()
    net.load_state_dict(checkpoint['model'])
    if n_gpu > 1:
        net = torch.nn.DataParallel(net, range(n_gpu))

    return net, checkpoint['model']


def train(num_episode, agent, env, output):
    T = []  # trajectory

    for cur_episode in range(num_episode):
        # reset as it is the start of an episode
        state = deepcopy(env.reset())
        print('cur_episode: {}'.format(cur_episode))
        #if args.show:
            #for i in range(100):
            #    print(agent.random_action())        
            #return

        while True:
            # agent pick action
            if cur_episode < args.warmup:   
                action = agent.random_action()
                if args.show: print('action: {}'.format(action))               
                    
            else:
                action = agent.select_action(state)            
                if args.show: print('select action: {}'.format(action))


            # env response with next_observation, reward, terminate_info
            next_state, reward, done, info = env.step(action, args.show)
            T.append([deepcopy(state), action, done])

            state = deepcopy(next_state)

            if done:    # end of episode
                episode_reward = reward

                print('# {}: episode_reward: {:.4f} loss: {:.4f} acc: {:.4f}, ratio: {:.4f} best_reward: {:.4f}\n'.format(
                        cur_episode, episode_reward, info['loss'], info['acc'], info['compress_ratio'], env.best_reward))

                # agent observe and update policy
                for s_t, a_t, done in T:
                    agent.append_replay(s_t, a_t, episode_reward, done)
                    if cur_episode >= args.warmup:
                        agent.update_policy()

                T = []

                tf_writer.add_scalar('reward/last', episode_reward, cur_episode)
                tf_writer.add_scalar('reward/best', env.best_reward, cur_episode)
                tf_writer.add_scalar('info/acc', info['acc'], cur_episode)
                tf_writer.add_scalar('info/comress_ratio', info['compress_ratio'], cur_episode)

                # log_writer.write('best reward: {}\n'.format(env.best_reward))
                # log_writer.write('best policy: {}\n'.format(env.best_strategy))
                break
        
        if ((cur_episode + 1) % args.save_internal == 0):
            agent.save_model((cur_episode + 1) // args.save_internal)        

    # agent.save_model()
    log_writer.close()


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    model, args.ckp = create_model(args.ckp, args.n_gpu)

    env = Env(model, args)

    if args.job == 'train':
        # build folder and log
        folder_name = 'tau{}'.format(args.tau)
        args.output = get_output_folder(args.output, folder_name)
        print('=> Saving logs to {}'.format(args.output))
        tf_writer = SummaryWriter(logdir=args.output)
        log_writer = open(os.path.join(args.output, 'log.txt'), 'w')
        env.set_log(log_writer)

        args.rmsize = args.rmsize * env.compressible_length
        n_state = env.layer_embedding.shape[1]
        agent = DDPG(n_state, log_writer, args)
        if args.resume != '':
            print('<= Loading weights from {}/actor_critic-{}.pkl'.format(args.resume, args.resumenum))
            agent.load_weights(args.resume, args.resumenum)

        train(args.episode, agent, env, args.output)
