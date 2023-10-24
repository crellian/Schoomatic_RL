import threading
import numpy as np
import random
import pygame
from pygame.locals import *

from singleCarla.carla_lap_env_ import CarlaLapEnv
from ray.rllib.policy.policy import Policy
import cv2
env_config ={
"addresses": [["192.168.0.183", 2345]
        ],
        "timeout": 10,  # this is in seconds
        "synchronous": False,
        "delta_seconds": -1,
        "fps": -1,
        "server_display": False,
        "debug_mod": False,

        "render_hud": True,
        "rgb_display": False, #if we want rgb_display
        "rgb_viewer_res": (1280, 720), #only if rgb_display is activated if rgb_display is True
        "bev_display": False, #if we want bev display
        "bev_viewer_res": (128, 128), #only if bev_display is activated
        "horizontal_fov": 80.0, #upto you
        "rgb_obs": False, #if we want the fpv observation
        "rgb_obs_res": (84, 84), #only is activated if rgb_obs is true
        "bev_obs": True, #if we want the bev observation
        "bev_obs_res": (42, 42), #only is activated if bev_res is true

        "task_config":
            {
                "max_timesteps": 2000,  # max timesteps for the episode
                "town": "Town05",
                "src_loc": (-67, -91),  # if its None generate randomly
                "dst_loc": (27, -72),  # if its None generate randomly
                "pedestrian_fq": 30.0, #0.0 -> 100.0, how many moving pedestrians on the curb and the crosswalk
                "vehicle_fq": 23.0, #0.0 -> 100.0, how many moving vehicles on the road
                "pedestrian_obstacle_fq": 0.0, #0.0 -> 100.0 how many static pedestrian obstacles in the scene
                "vehicle_obstacle_fq": 8.0, #0.0 -> 100.0 how many static vehicle obstacles in the scene
                "sparse_reward_fn": False, #if its false then implement the reward fn we talked about
                "goal_reward": "propdist", #goal reward proportional to the distance
                "goal_tolerance": 1,
                "terminate_reward": -1000.0, #add an assertion that this reward always has to be negative
                "sparse_config": #only is activated if sparse_reward_fn is True
                    {
                        "sparse_reward_thresh": 200.0,
                        "sparse_reward_fn": (lambda i: 200.0 + i*200.0),
                    },
                "curriculum": False,
                "cirriculum_config": #only is activated if curriculum is True
                    {
                        "num_goals": 1, #-1 if there is no limit
                        "start_dist": 300,
                        "end_dist": -1,
                        "cirriculum_fn": (lambda i: "start_dist" + 100*i), #where i is the index of the generation
                    }
            },
        "action_smoothing": 0,  # dont worry about this
    }


actions = [[0, -1],
                [0.5, -0.2],
                [1.0, 0],
                [0.5, 0.2],
                [0, 1]
                ]

if __name__ == '__main__':
# Example of using CarlaEnv with keyboard controls
    env = CarlaLapEnv(env_config)

    while True:
        while True:
            try:
                obs = env.reset()
            except RuntimeError as e:  # disconnected from the server
                print(e)
                env.init_world()  # internal loop until reconnected
            else:
                break
        reward = 0
        a = 0
        obs = env.observation
        obs = np.concatenate((obs['obs'], obs['aux']), axis=-1)
        action = np.zeros(2)
        while True:
            a = random.randint(0, 4)
            # Take action
            t = threading.Thread(target=env.step, args=([1,0],))
            t.start()
            t.join(10)
            if t.is_alive():
                print("Client lost connection to carla server, reset the client..")
                break
            env.render()
            obs, reward, done, info = env.observation, env.last_reward, env.terminal_state, env.info
            obs = np.concatenate((obs['obs'], obs['aux']), axis=-1)

            if done:
                break
