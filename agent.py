import gymnasium as gym

from stable_baselines3 import DQN

from gym_examples.envs.slice_creation_env4 import SliceCreationEnv4

from os import rename

env = SliceCreationEnv4()


model = DQN.load("dqn_slices4(Arch:16; learn:1e-3; starts:250k; fraction:0_5; train: 1.5M).zip", env)
#model = DQN.load("/home/mario/Documents/DQN_Models/Model 1/gym-examples2/dqn_slices", env)

obs, info = env.reset()

cont = 0

for i in range(50):
    while cont<99:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print('Action: ', action,'Observation: ', obs, ' | Reward: ', reward, ' | Terminated: ', terminated)
        cont += 1
        if terminated or truncated:
            obs, info = env.reset()
    cont = 0
    # Comment after training of Model 2
    rename('Global_Parameters.db','Global_Parameters{}.db'.format(str(i+1)))
    obs, info = env.reset()