import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from core import MLPActorCritic

env = gym.make('HalfCheetah-v4', render_mode="human")
ac = MLPActorCritic(env.observation_space, env.action_space)
ac.load_state_dict(torch.load("model.pth"))

test_epoch = 10
local_steps_per_epoch = 4000

o, _ = env.reset()

for epoch in range(test_epoch):
    for t in range(local_steps_per_epoch):
        o = torch.as_tensor(o, dtype=torch.float32)
        
        a, v, logp = ac.step(o)

        next_o, r, d, trunc, _ = env.step(a)
        o = next_o
        env.render()