
import torch
import torch.nn as nn
import numpy as np
import random

from torch.optim import Adam
from scipy import stats
from rl.memory import SequentialMemory
from utils import to_tensor, to_numpy

criterion = nn.MSELoss()

class Actor(nn.Module):
    def __init__(self, n_state, n_action, hidden1=400, hidden2=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_state, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_action)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


class Critic(nn.Module):
    def __init__(self, n_state, n_action, hidden1=400, hidden2=300):
        super(Critic, self).__init__()
        self.fc1_state = nn.Linear(n_state, hidden1)
        self.fc1_action = nn.Linear(1, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, xs):
        s, a = xs
        out = self.fc1_state(s) + self.fc1_action(a)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class DDPG(object):
    def __init__(self, n_state, log_writer, args):
        self.n_state = n_state
        self.log_writer = log_writer
        self.output = args.output

        self.action_start = args.action_start
        self.action_end = args.action_end
        self.n_action = self.action_end - self.action_start + 1        

        # create actor and critic network
        net_config = {'n_state': self.n_state, 'n_action': self.n_action,
                      'hidden1': args.hidden1, 'hidden2': args.hidden2}

        self.actor = Actor(**net_config)
        self.actor_target = Actor(**net_config)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_a)
        self.critic = Critic(**net_config)
        self.critic_target = Critic(**net_config)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_c)

        # make sure target is with the same weight
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        # create replay buffer
        self.memory = SequentialMemory(size=args.rmsize)

        # hyper-parameters
        self.batch_size = args.bsize
        self.discount = args.discount
        self.tau = args.tau

        # noise ???
        '''
        
        '''

        if torch.cuda.is_available():
            self.cuda()

        # moving average baseline
        self.moving_average = None
        self.moving_alpha = args.moving_alpha

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def random_action(self):
        # print('random_int')
        return random.randint(self.action_start, self.action_end)

    def select_action(self, state):
        action_prob = to_numpy(self.actor(to_tensor(state.reshape(1, -1)))).squeeze(0)
        dice = stats.rv_discrete(values=(range(self.action_start, self.action_end + 1), action_prob))
        action = dice.rvs(size=1)    
    
        # print(action_prob)
        # print('select action: {}'.format(action[0]))
        return action[0]

    def get_exact_action(self, state_batch, kind):
        if kind == 0:
            action_prob = self.actor_target(state_batch)
        else:
            action_prob = self.actor(state_batch)

        max_val, prediction = torch.max(action_prob, 1)
        prediction = prediction.reshape(self.batch_size, -1).float()
        return prediction / self.n_action

    def update_policy(self):
        # sample batch
        # print('start update policy\n')
        # self.log_writer.flush()

        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        action_batch = (action_batch - self.action_start) / self.n_action

        # normalize the reward
        batch_mean_reward = reward_batch.mean().item()
        if self.moving_average is None:
            self.moving_average = batch_mean_reward
        else:
            self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
        reward_batch -= self.moving_average

        # update critic
        self.critic.zero_grad()

        q_batch = self.critic([state_batch, action_batch])

        with torch.no_grad():   # prepare for the target q batch
            next_q_values = self.critic_target([next_state_batch,
                                                self.get_exact_action(next_state_batch, 0)])
        target_q_batch = reward_batch + self.discount * terminal_batch * next_q_values

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # update actor
        self.actor.zero_grad()

        policy_loss = -self.critic([state_batch, self.get_exact_action(state_batch, 1)])
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()
        
        # print('end update policy\n')
        # self.log_writer.flush()
        
        # target network update
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def hard_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )

    def append_replay(self, s_t, a_t, r_t, done):
        self.memory.append(s_t, a_t, r_t, done)

    def save_model(self, num):
        torch.save(self.actor.state_dict(), '{}/actor-{}.pkl'.format(self.output, num))
        torch.save(self.critic.state_dict(), '{}/critic-{}.pkl'.format(self.output, num))

    def load_weights(self, state_dir, num):
        self.actor.load_state_dict(torch.load('{}/actor-{}.pkl'.format(state_dir, num)))
        self.critic.load_state_dict(torch.load('{}/critic-{}.pkl'.format(state_dir, num)))
        self.actor_target.load_state_dict(torch.load('{}/actor-{}.pkl'.format(state_dir, num)))
        self.critic_target.load_state_dict(torch.load('{}/critic-{}.pkl'.format(state_dir, num)))
    
