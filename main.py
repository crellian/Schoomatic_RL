import threading
import numpy as np
import random
import pygame
from pygame.locals import *
from singleCarla.getmap import Map
from singleCarla.carla_lap_env import CarlaLapEnv
from ray.rllib.policy.policy import Policy
import cv2
env_config ={
"addresses": [["192.168.0.21", 2066]
        ],
        "timeout": 10,  # this is in seconds
        "synchronous": False,
        "delta_seconds": -1,
        "fps": -1,
        "server_display": False,
        "debug_mod": False,

        "render_hud": True,
        "rgb_display": True, #if we want rgb_display
        "rgb_viewer_res": (1280, 720), #only if rgb_display is activated if rgb_display is True
        "bev_display": False, #if we want bev display
        "bev_viewer_res": (128, 128), #only if bev_display is activated
        "horizontal_fov": 80.0, #upto you
        "rgb_obs": True, #if we want the fpv observation
        "rgb_obs_res": (84, 84), #only is activated if rgb_obs is true
        "bev_obs": True, #if we want the bev observation
        "bev_obs_res": (64, 64), #only is activated if bev_res is true

        "task_config":
            {
                "max_timesteps": 3000,  # max timesteps for the episode
                "town": "Town05",
                "src_loc": None,  # if its None generate randomly
                "dst_loc": None,  # if its None generate randomly
                "pedestrian_fq": 30.0, #0.0 -> 100.0, how many moving pedestrians on the curb and the crosswalk
                "vehicle_fq": 23.0, #0.0 -> 100.0, how many moving vehicles on the road
                "pedestrian_obstacle_fq": 0.0, #0.0 -> 100.0 how many static pedestrian obstacles in the scene
                "vehicle_obstacle_fq": 8.0, #0.0 -> 100.0 how many static vehicle obstacles in the scene
                "sparse_reward_fn": False, #if its false then implement the reward fn we talked about
                "goal_reward": "propdist", #goal reward proportional to the distance
                "goal_tolerance": 5,
                "terminate_reward": -1000.0, #add an assertion that this reward always has to be negative
                "resolution": 5,
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
                       [0.4, -0.2],
                       [0.75, 0],
                       [0.2, 0],
                       [0.0, 0],
                       [0.4, 0.2],
                       [0, 1]
                       ]

model_path = "/home2/carla/checkpoint"
if __name__ == '__main__':
# Example of using CarlaEnv with keyboard controls
    env = CarlaLapEnv(env_config)
    action = np.zeros(2)
    my_restored_policy = Policy.from_checkpoint(model_path)
    # init_state = state = [np.zeros([256], np.float32) for _ in range(2)]
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    cnt = 0
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
        init_state = state = my_restored_policy.get_initial_state()
        getmap = Map()
        cnt += 1
        out = cv2.VideoWriter(str(cnt) + "_rllib.mp4", fourcc, 100.0, (1280, 720))

        while True:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[K_LEFT] or keys[K_a]:
                action[1] = -0.8
            elif keys[K_RIGHT] or keys[K_d]:
                action[1] = 0.8
            elif keys[K_s]:
                im = cv2.cvtColor(info["rgb_obs"], cv2.COLOR_BGR2RGB)
                cv2.imwrite("1_.jpg", im)
                cv2.imwrite("1.jpg", obs[:, :, 0])
            else:
                action[1] = 0.0
            action[1] = np.clip(action[1], -1, 1)
            action[0] = 0.8 if keys[K_UP] or keys[K_w] else 0.0


            a, state, _ = my_restored_policy.compute_single_action(obs, state, prev_action=a, prev_reward=reward)
            # Take action
            t = threading.Thread(target=env.step, args=(action,))
            t.start()
            t.join(10)
            if t.is_alive():
                print("Client lost connection to carla server, reset the client..")
                break

            obs, reward, done, info = env.observation, env.last_reward, env.terminal_state, env.info
            l_t, r, r_ = getmap.method(action, info)

            env.render(r)
            tmp = obs["obs"]
            obs["obs"] = r

            out_img = env.viewer_image['rgb']
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            r = cv2.resize(r, (200, 200), interpolation=cv2.INTER_LINEAR)
            r = cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)
            r[0, :] = (150, 146, 135)
            r[-1, :] = (150, 146, 135)
            r[:, 0] = (150, 146, 135)
            r[:, -1] = (150, 146, 135)
            gt = cv2.resize(tmp, (200, 200), interpolation=cv2.INTER_LINEAR)
            gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
            gt[0, :] = (150, 146, 135)
            gt[-1, :] = (150, 146, 135)
            gt[:, 0] = (150, 146, 135)
            gt[:, -1] = (150, 146, 135)
            RGB_img = cv2.resize(r_, (200, 200), interpolation=cv2.INTER_LINEAR)
            RGB_img = cv2.cvtColor(RGB_img, cv2.COLOR_GRAY2RGB)
            RGB_img[0, :] = (150, 146, 135)
            RGB_img[-1, :] = (150, 146, 135)
            RGB_img[:, 0] = (150, 146, 135)
            RGB_img[:, -1] = (150, 146, 135)
            out_img[50:50 + 200, 100:100 + 200] = gt
            out_img[270:270 + 200, 100:100 + 200] = RGB_img
            out_img[490:490 + 200, 100:100 + 200] = r
            out.write(out_img)
            print(env.step_count)
            if done:
                break
        out.release()

    cap.release()

