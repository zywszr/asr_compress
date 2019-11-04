
import random
import warnings

import numpy as np

from collections import namedtuple
from utils import *


Experience = namedtuple('Experience', 'state, action, reward, next_state, terminal')


class RingBuffer(object):
    def __init__(self, size):
        self.size = size
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(size)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.size]

    def append(self, v):
        if self.length < self.size:
            self.length += 1
        else:   # length == size
            self.start = (self.start + 1) % self.size
        self.data[(self.start + self.length - 1) % self.size] = v


class SequentialMemory(object):
    def __init__(self, size):
        self.size = size

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.states = RingBuffer(size)
        self.actions = RingBuffer(size)
        self.rewards = RingBuffer(size)
        self.terminals = RingBuffer(size)

    def sample_batch_indexes(self, low, high, size):
        # sample indexes from [low, high)
        if high - low >= size:
            batch_idxs = random.sample(range(low, high), size=size)
        else:
            warnings.warn(
                'Not enough entries to sample without replacement. '
                'Consider increasing your warm-up phase to avoid oversampling!')
            batch_idxs = np.random.randint(low, high, size=size)

        return batch_idxs

    def sample_batch(self, batch_size):
        batch_idxs = self.sample_batch_indexes(0, self.n_entries - 1, batch_size)

        experiences = []
        for idx in batch_idxs:
            state = self.states[idx]
            action = self.actions[idx]
            reward = self.rewards[idx]
            next_state = self.states[idx + 1]
            terminal = self.states[idx]
            experiences.append(Experience(state=state, action=action, reward=reward,
                                          next_state=next_state, terminal=terminal))

        return experiences

    def sample_and_split(self, batch_size):
        experiences = self.sample_batch(batch_size)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        for e in experiences:
            state_batch.append(e.state)
            action_batch.append(e.action)
            reward_batch.append(e.reward)
            next_state_batch.append(e.next_state)
            terminal_batch.append(0. if e.terminal else 1.)

        state_batch = to_tensor(state_batch).reshape(batch_size, -1)
        action_batch = to_tensor(action_batch).reshape(batch_size, -1)
        reward_batch = to_tensor(reward_batch).reshape(batch_size, -1)
        next_state_batch = to_tensor(next_state_batch).reshape(batch_size, -1)
        terminal_batch = to_tensor(terminal_batch).reshape(batch_size, -1)

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def append(self, state, action, reward, terminal):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    @property
    def n_entries(self):
        return len(self.states)
