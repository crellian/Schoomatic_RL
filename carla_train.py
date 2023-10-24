import sys
import yaml
import threading
import gym
import ray

from ray.rllib.algorithms.ppo import PPOConfig
from single_cnn import SingleTorchModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

from ray.rllib.algorithms.ppo import PPO
from ray.tune.logger import pretty_print

from singleCarla.carla_lap_env import CarlaLapEnv

class SingleCarlaEnv(gym.Env):
    def __init__(self, env_config):
        self.env = CarlaLapEnv(env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.actions = [[0, -1],
                       [0.2, -0.5],
                       [0.5, -0.5],
                       [0.75, -0.2],
                       [0.5, 0],
                       [1.0, 0],
                       [0.75, 0.2],
                       [0.5, 0.5],
                       [0.2, 0.5],
                       [0, 1]
                       ]

    def reset(self, **kwargs):
        while True:
            try:
                self.env.reset(**kwargs)
            except RuntimeError as e:  # disconnected from the server
                print(e)
                self.env.init_world()  # internal loop until reconnected
            else:
                break
        return self.env.observation

    def step(self, a):
        t = threading.Thread(target=self.env.step, args=(self.actions[a],))
        t.start()
        t.join(10)
        if t.is_alive():
            self.env.terminal_state = True
            self.env.info["discard"] = "Client lost connection to carla server"       # TODO
            print("Client lost connection to carla server, reset the client..")
        self.env.render()
        return self.env.observation, self.env.last_reward, self.env.terminal_state, self.env.info


if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("mod", SingleTorchModel)

    with open(sys.argv[1]) as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)

    config = (
        PPOConfig()
        .environment(SingleCarlaEnv, env_config=env_config)
        .framework("torch")
        .rollouts(num_rollout_workers=len(env_config["addresses"]), preprocessor_pref="rllib")
        .training(
            model={
            "custom_model": "mod",
            "framestack": False,
            "use_lstm": True,
            "vf_share_layers" : True},
        lambda_=0.95,
        lr=0.0001,
        kl_coeff=0.5,
        clip_param=0.1,
        vf_clip_param=10.0,
        grad_clip=100.0,
        entropy_coeff=0.01,
        train_batch_size=5000,
        sgd_minibatch_size=500,
        num_sgd_iter=10)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=.3,
            num_gpus_per_worker=.12,
            num_cpus_per_worker=1)
        # .callbacks(MyCallbacks)
    )

    algo = PPO(config=config)
    for _ in range(1000):
        result = algo.train()
        print(pretty_print(result))
    algo.stop()

    ray.shutdown()
