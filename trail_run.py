import gym
import highway_env
import random
from collections import deque
from deepnetwork import DeepQnetwork
from config import *
from tqdm import tqdm
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join("tmp",'output2.avi'),fourcc, 20.0, (IM_W,IM_H))

class Highway_GUI:

    def __init__(self):
        self.dqn = DeepQnetwork(training_mode=False)
        self.dqn.load_model()
        self.dqn.epsilon = 0
        self.previous_memory = deque(maxlen=node_history_size)
        self.env = gym.make('highway-v0')
        self.set_properties()

    def set_properties(self):
        '''#self.env.config["offscreen_rendering"] = True
        self.env.config["vehicles_count"] = n_vehicles-1
        #self.env.config["features"] = ["presence", "x", "y", "vx", "vy"]
        self.env.config["lanes_count"] = 4
        self.env.config["screen_width"] = IM_W
        self.env.config["screen_height"] = IM_H
        self.env.config["scaling"] = 3.0
        self.env.config['reward_speed_range'] = [30, 70]
        self.env.config['centering_position'] = [0.5, 0.5]

        '''
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
            "lanes_count":4
        }
        self.env.configure(configr)

    def get_batch(self, sampling_size):
        this_batch = random.sample(self.previous_memory, sampling_size)
        current_nodes, actions, next_nodes, rewards = list(zip(*this_batch))
        return [np.stack(current_nodes), np.array(actions), np.stack(next_nodes), np.array(rewards)]

    def train_network(self):
        data = self.get_batch(batch_size)
        self.dqn.train(data)

    def process_observation(self, obs):
        obs = obs[:n_vehicles, 1:]
        return obs

    def get_reward(self, observation, info):
        if info['crashed']:
            reward = -1
        else:
            if np.sum(observation[1:, 1]) > 0:
                reward = 0
            else:
                reward = 5
        return reward

    def run(self, episodes, train_frequency=2):

        for episode in tqdm(range(episodes)):
            self.observation = self.env.reset()
            #self.observation = self.env.render(mode='rgb_array')
            reward_history = []
            step_counter = 0
            while 1:
                action = self.dqn.get_action(self.observation)
                self.next_observation, reward, done, info = self.env.step(action)
                #self.next_obs = self.process_observation(self.next_obs)
                #reward = self.get_reward(self.next_obs, info)
                reward_history.append(reward)
                frame = self.env.render(mode='rgb_array')
                self.previous_memory.append([self.observation, action, self.next_observation, reward])
                self.observation = self.next_observation
                '''if  step_counter%3 == 0:
                    if len(self.previous_memory) >= batch_size:
                        self.train_network()'''
                out.write(frame)
                if done:
                    break
            '''
            self.dqn.save_log(episode, np.mean(reward_history), "episodic_reward.csv")
            if episodes//2 >= episode >=1:
                new_epsilon = self.dqn.epsilon-self.dqn.decay
                self.dqn.epsilon = max(new_epsilon, self.dqn.min_epsilon)
            if episode>10 and episode%10 == 0:
                self.dqn.update_prediction_network()
            '''
        cap.release()
        out.release()
        cv2.destroyAllWindows()

h = Highway_GUI()
h.run(episodes=100)
