import gym
import highway_env
import random
from config import *
env = gym.make('highway-v0')
screen_width, screen_height = IM_W, IM_H

configr = {
            "offscreen_rendering": False,
            "observation": {
                "type": "GrayscaleObservation",
                "weights": [0.9, 0.1, 0.5],  # weights for RGB conversion
                "stack_size": 4,
                "observation_shape": (screen_width, screen_height)
            },
            "screen_width": screen_width,
            "screen_height": screen_height,
            "scaling": 5.75,
            "vehicles_count":n_vehicles-1,

            "policy_frequency": 2,
            "lanes_count":4
        }
env.configure(configr)
env.reset()
for _ in range(500):
    action = random.choice(list(range(0,5)))
    obs, reward, done, info = env.step(action)
    if done:
        break
    print(obs,reward)
    env.render()
