import torch
import psutil
import math
import pickle
import transforms3d
import cv2
import carla
import gym
import pygame
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
import torch.nn as nn
from .hud import HUD
from .planner import compute_route_waypoints
from .wrappers import *
from .models import VAEBEV

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")

MAX_NPC_PEDESTRIAN = 200
MAX_NPC_VEHICLE = 200

def is_used(port):
    """Checks whether or not a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]

def distance(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

class CarlaLapEnv(gym.Env):
    def __init__(self, env_config):
        self.host = env_config["addresses"][0][0]
        self.port = int(env_config["addresses"][0][1])
        print("host: "+self.host)
        print("port: "+str(self.port))
        if env_config["debug_mod"] and int(env_config.worker_index) == len(env_config["addresses"]):
            # enable debug mode on the last worker
            self.debug = True
            self.reward_ls = []
            self.observation_ls = []
            self.terminal_state_ls = []
            self.info_ls = []
            self.count = 0
        else:
            self.debug = False

        self.timeout = env_config["timeout"]
        self.display = env_config["server_display"]
        self.synchronous = env_config["synchronous"]
        self.fps = env_config["fps"]
        self.delta_seconds = env_config["delta_seconds"]

        self.render_hud = env_config["render_hud"]
        self.rgb_display = env_config["rgb_display"]
        self.rgb_viewer_res = env_config["rgb_viewer_res"]
        self.bev_display = env_config["bev_display"]
        self.bev_viewer_res = env_config["bev_viewer_res"]
        self.rgb_obs = env_config["rgb_obs"]
        self.rgb_obs_res = env_config["rgb_obs_res"]
        self.bev_obs = env_config["bev_obs"]
        self.bev_obs_res = env_config["bev_obs_res"]
        self.horizontal_fov = env_config["horizontal_fov"]

        self.action_smoothing = env_config["action_smoothing"]

        self.town = env_config["task_config"]["town"]
        self.max_timesteps = env_config["task_config"]["max_timesteps"]
        self.sparse_reward_fn = env_config["task_config"]["sparse_reward_fn"]
        self.goal_reward = env_config["task_config"]["goal_reward"]
        self.goal_tolerance = env_config["task_config"]["goal_tolerance"]
        self.terminate_reward = env_config["task_config"]["terminate_reward"]
        self.src_loc = env_config["task_config"]["src_loc"]
        self.dst_loc = env_config["task_config"]["dst_loc"]
        self.resolution = env_config["task_config"]["resolution"]

        self.pedestrian_fq = env_config["task_config"]["pedestrian_fq"]
        self.vehicle_fq = env_config["task_config"]["vehicle_fq"]
        self.pedestrian_obstacle_fq = env_config["task_config"]["pedestrian_obstacle_fq"]
        self.vehicle_obstacle_fq = env_config["task_config"]["vehicle_obstacle_fq"]

        # Setup gym environment
        self.action_space = gym.spaces.Discrete(7)
        if self.bev_obs:
            self.observation_space = gym.spaces.Dict(
                {"obs": gym.spaces.Box(low=0.0, high=255.0, shape=(*self.bev_obs_res,), dtype=np.float32),
                "aux": gym.spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32)})
        elif self.rgb_obs:
            self.observation_space = gym.spaces.Dict(
                #{"obs": gym.spaces.Box(low=0.0, high=255.0, shape=(*self.rgb_obs_res, 3), dtype=np.float32),
                {"obs": gym.spaces.Box(-10, 10, shape=(512,), dtype=np.float32),
                "aux": gym.spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32)})

        self.world = None
        self.vehicle = None
        self.camera = None
        self.dashcam = None
        self.start_wp = None
        self.goal_wp = None
        self.route_waypoints = None

        self.init_world()

        if self.bev_obs:
            vae_model_path = "/home2/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
            self.vae = VAEBEV(channel_in=1, ch=16, z=32).to(device)
            vae_ckpt = torch.load(vae_model_path, map_location="cpu")
            self.vae.load_state_dict(vae_ckpt['model_state_dict'])
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False

        elif self.rgb_obs:
            self.resnet = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_ftrs, 512).to(device)
            self.resnet.eval()
            for param in self.resnet.parameters():
                param.requires_grad = False

    def init_world(self):
        while True:
            try:
                # Connect to carla
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(self.timeout)
                # Create world wrapper
                self.world = World(self.client, self.town)

                settings = self.world.get_settings()
                settings.synchronous_mode = self.synchronous
                if self.rgb_obs:
                    settings.no_rendering_mode = False
                else:
                    settings.no_rendering_mode = True

                if self.delta_seconds > 0:
                    settings.fixed_delta_seconds = self.delta_seconds
                else:
                    settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)

                # Create hud
                # Initialize pygame for visualization
                self.clock = pygame.time.Clock()
                if self.render_hud:
                    pygame.init()
                    pygame.font.init()
                    rgb_viewer_width, rgb_viewer_height = self.rgb_viewer_res
                    bev_viewer_width, bev_viewer_height = self.bev_viewer_res
                    width, height = rgb_viewer_width + bev_viewer_width, rgb_viewer_height + bev_viewer_height
                    self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
                    self.hud = HUD(width, height)
                    self.world.on_tick(self.hud.on_world_tick)
                else:
                    self.display = None
                    self.hud = None

                # Spawn obstacles
                # self.actors = self._spawn_npcs(int(self.vehicle_fq / 100 * MAX_NPC_VEHICLE),
                #             int(self.vehicle_obstacle_fq / 100 * MAX_NPC_VEHICLE),
                #             int(self.pedestrian_fq / 100 * MAX_NPC_PEDESTRIAN),
                #             int(self.pedestrian_obstacle_fq / 100 * MAX_NPC_PEDESTRIAN))

            except Exception as e:
                print(e)
                print("time-out of "+str(self.timeout)+"000ms while waiting for the simulator, make sure "
                                                       "the simulator is ready and connected to", self.host, ":", self.port)
            else:
                break

    def reset(self, **kwargs):
        if self.world is not None:
            self.world.destroy()

        # Get destination waypoint
        if self.dst_loc:
            self.goal_wp = self.world.map.get_waypoint(carla.Location(x=self.dst_loc[0], y=self.dst_loc[1]))
        else:
            spawn_points = self.world.map.get_spawn_points()
            random.shuffle(spawn_points, random.random)
            self.goal_wp = self.world.map.get_waypoint(spawn_points[0].location)

        # Create vehicle and attach sensor to it
        # Generate waypoints along the lap
        if self.src_loc:
            self.route_waypoints = compute_route_waypoints(self.world.map,
                                                           self.world.map.get_waypoint(carla.Location(x=self.src_loc[0], y=self.src_loc[1])),
                                                           self.goal_wp,
                                                           resolution=self.resolution)
            print(len(self.route_waypoints))
            while True:  # until vehicle successfully spawned
                #start_i = random.randint(0, int(len(self.route_waypoints) / 2))  # spawn at a random point on the path
                self.start_wp = self.route_waypoints[0][0]
                spawn_transform = self.start_wp.transform
                spawn_transform.location.z += 1

                self.vehicle = Vehicle(self.world, spawn_transform,
                                       on_collision_fn=lambda e: self._on_collision(e),
                                       vehicle_type="scoomatic.scoomatic.uni_a")

                if self.vehicle.actor is not None:
                    self.route_waypoints = self.route_waypoints[1:]
                    break

        else:
            spawn_transforms = self.world.map.get_spawn_points()
            random.shuffle(spawn_transforms, random.random)

            for i in range(0, len(spawn_transforms)):
                next_spawn_transform = spawn_transforms[i % len(spawn_transforms)]
                next_spawn_transform.location.z += 1
                vehicle = Vehicle(self.world, next_spawn_transform,
                                  on_collision_fn=lambda e: self._on_collision(e),
                                   vehicle_type="scoomatic.scoomatic.uni_a")
                if vehicle.actor is not None:
                    self.vehicle = vehicle
                    self.start_wp = self.world.map.get_waypoint(next_spawn_transform.location)
                    self.route_waypoints = compute_route_waypoints(self.world.map,
                                                                   self.start_wp,
                                                                   self.goal_wp,
                                                                   resolution=self.resolution)
                    self.route_waypoints = self.route_waypoints[1:]

                    print("Spawned actor \"{}\"".format(vehicle.actor.type_id))
                    break
                else:
                    print("Could not spawn hero, changing spawn point")

        if self.hud:
            self.hud.set_vehicle(self.vehicle)


        rgb_viewer_width, rgb_viewer_height = self.rgb_viewer_res
        bev_viewer_width, bev_viewer_height = self.bev_viewer_res
        rgb_obs_width, rgb_obs_height = self.rgb_obs_res
        bev_obs_width, bev_obs_height = self.bev_obs_res
        # Create cameras
        if self.rgb_display:
            self.camera = Camera(self.world, rgb_viewer_width, rgb_viewer_height,
                                 transform=camera_transforms["spectator"], fov=self.horizontal_fov,
                                 attach_to=self.vehicle, on_recv_image=lambda e: self._set_rgb_viewer_image(e),
                                 sensor_tick=0.0)
        if self.rgb_obs:
            self.dashcam = Camera(self.world, rgb_obs_width, rgb_obs_height,
                                  transform=camera_transforms["dashboard"], fov=self.horizontal_fov,
                                  attach_to=self.vehicle,
                                  on_recv_image=lambda e: self._set_rgb_observation_image(e),
                                  sensor_tick=0.0)

        if self.bev_display:
            self.birdview_producer_display = BirdViewProducer(
                self.client,  # carla.Client
                target_size=PixelDimensions(width=bev_viewer_width, height=bev_viewer_height),
                pixels_per_meter=2.5,
                crop_type=BirdViewCropType.FRONT_AREA_ONLY,
                render_lanes_on_junctions=False
            )
        else:
            self.birdview_producer_display = None
        if self.bev_obs:
            self.birdview_producer_obs = BirdViewProducer(
                self.client,  # carla.Client
                target_size=PixelDimensions(width=bev_obs_width, height=bev_obs_height),
                pixels_per_meter=2.5,
                crop_type=BirdViewCropType.FRONT_AREA_ONLY,
                render_lanes_on_junctions=False
            )
        else:
            self.birdview_producer_obs = None


        self.terminal_state = False  # Set to True when we want to end episode
        self.extra_info = []  # List of extra info shown on the HUD
        self.observation = {}
        self.observation_buffer = None  # Last received observation
        self.viewer_image = {}
        self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.step_count = 0
        self.collision = False
        self.last_reward = 0
        self.info = {}

        # Metrics
        self.previous_location = self.vehicle.get_transform().location

        # DEBUG: Draw path
        #self._draw_path(life_time=1000.0, skip=10)

        self.step(None)
        return


    def render(self, r, mode="human"):
        if self.display:
            # Add metrics to HUD
            self.extra_info.extend([
                "Reward: % 19.2f" % self.last_reward
            ])

            # Blit image from spectator camera
            self.display.blit(pygame.surfarray.make_surface(self.viewer_image['bev'].swapaxes(0, 1)), (0, 0))
            self.display.blit(pygame.surfarray.make_surface(self.viewer_image['rgb'].swapaxes(0, 1)), (self.bev_viewer_res[0],0))
            self.display.blit(pygame.surfarray.make_surface(r.swapaxes(0, 1)), (0, 400))            # Superimpose current observation into top-right corner



            # Render HUD
            self.hud.render(self.display)

            self.extra_info = []  # Reset extra info list

            # Render to screen
            pygame.display.flip()
            #self._draw_path(life_time=1)

    def step(self, action):
        # Asynchronous update logic
        if not self.synchronous:
            if self.fps <= 0:
                # Go as fast as possible
                self.clock.tick()
            else:
                # Sleep to keep a steady fps
                self.clock.tick_busy_loop(self.fps)

        # Take action
        if action is not None and self.step_count % 2 == 0:     # control signal fps is half of the camera
            throttle, steer = [float(a) for a in action]
            self.vehicle.steer = self.vehicle.steer * self.action_smoothing + steer * (1.0 - self.action_smoothing)
            self.vehicle.throttle = self.vehicle.throttle * self.action_smoothing + throttle * (
                        1.0 - self.action_smoothing)
            self.vehicle.control.left_velocity = (self.vehicle.throttle + self.vehicle.steer * 4) * 1000
            self.vehicle.control.right_velocity = (self.vehicle.throttle - self.vehicle.steer * 4) * 1000

            self.world.tick()

        if self.hud:
            self.hud.tick(self.world, self.clock)

        # Synchronous update logic
        if self.synchronous:
            self.clock.tick()
            while True:
                try:
                    self.world.wait_for_tick(seconds=1.0/self.fps + 0.1)
                    break
                except:
                    # Timeouts happen occasionally for some reason, however, they seem to be fine to ignore
                    self.world.tick()

        # Get most recent observation and viewer image
        observation = self._get_observation()
        if self.bev_obs:
            self.observation["obs"] = observation['bev'][:,:,0]
        elif self.rgb_obs:
            self.observation["obs"] = observation['rgb']
        self.info["rgb_obs"] = observation['rgb']
        self.viewer_image = self._get_viewer_image()

        # DEBUG: Draw current waypoint
        # self.world.debug.draw_point(self.route_waypoints[1][0].transform.location, 1, color=carla.Color(255, 0, 0), life_time=1.0)

        # Call reward fn
        self.step_count += 1
        self.last_reward = self._reward_fn()

        if self.debug:
            self.reward_ls.append(self.last_reward)
            self.observation_ls.append(self.observation)
            self.terminal_state_ls.append(self.terminal_state)
            self.info_ls.append(self.info)
            if self.terminal_state:
                dict = {"reward": self.reward_ls, "observation": self.observation_ls, "termination": self.terminal_state_ls, "info": self.info_ls}
                with open('debug/'+str(self.count)+'.pkl', 'wb') as f:
                    pickle.dump(dict, f)
                self.reward_ls = []
                self.observation_ls = []
                self.terminal_state_ls = []
                self.info_ls = []
                self.count += 1

        return

    def _transform(self, wp_transform):
        '''
        transform = self.vehicle.get_transform()
        location = transform.location
        waypoint = np.array([wp_loc.x, wp_loc.y, 0.0])
        vehicle = np.array([location.x, location.y, 0.0])
        loc = waypoint - vehicle

        R = transforms3d.euler.euler2mat(0, 0, math.radians(transform.rotation.yaw)).T
        loc_relative = np.dot(R, loc)
        #self.observation["aux"] = (loc_relative / np.linalg.norm(loc_relative))[:2]
        self.observation["aux"] = np.arctan2(loc[1], loc[0])
        '''
        def norm(rad):
            if rad < -math.pi:
                rad += 2*math.pi
            elif rad > math.pi:
                rad -= 2*math.pi

            rad = rad / math.pi
            return rad
        transform = self.vehicle.get_transform()
        location = transform.location
        orientation = transform.rotation.yaw

        wp_location = wp_transform.location
        dis_v = np.arctan2(wp_location.y-location.y, wp_location.x-location.x)
        diff = norm(np.radians(orientation) - dis_v)

        self.observation["aux"] = np.expand_dims(diff, axis=0)
    
    def _reward_fn(self):
        """Computes the reward"""
        hero = self.vehicle
        hero_loc = hero.get_location()

        if self.goal_reward == "propdist":
            if self.sparse_reward_fn:
                wp_transform = self.route_waypoints[-1][0].transform
                wp_loc = self.route_waypoints[-1][0].transform.location
            else:
                wp_transform = self.route_waypoints[1][0].transform
                wp_loc = self.route_waypoints[1][0].transform.location

        self._transform(wp_transform)
        #self._transform(self.goal_wp.transform.location)

        # success
        if distance(wp_loc, hero_loc) < self.goal_tolerance:
            if len(self.route_waypoints) > 1:
                self.route_waypoints = self.route_waypoints[1:]
            else:
                print("Done success")
                self.terminal_state = True
            reward = self.resolution
        else:
            reward = 0

        # fail
        if self.collision:
            print("Done collision")
            reward += self.terminate_reward
            self.terminal_state = True
        if self.step_count > self.max_timesteps:
            print("Done timeout")
            reward += self.terminate_reward
            self.terminal_state = True
        if self.step_count % 200 == 0:
            if distance(self.previous_location, hero_loc) < 0.1:
                print("Done idle")
                reward += self.terminate_reward
                self.terminal_state = True
            else:
                self.previous_location = hero_loc

        return reward

    def _draw_path(self, life_time=60.0, skip=0):
        """
            Draw a connected path from start of route to end.
            Green node = start
            Red node   = point along path
            Blue node  = destination
        """
        w0 = self.route_waypoints[1][0]
        self.world.debug.draw_point(
            w0.transform.location + carla.Location(z=0.1), 0.1,
            carla.Color(0, 255, 0),
            life_time, False)
        self.world.debug.draw_point(
            self.route_waypoints[-1][0].transform.location + carla.Location(z=0.25), 0.1,
            carla.Color(0, 0, 255),
            life_time, False)

    def _get_observation(self):
        if self.birdview_producer_obs:
            bev_obs = np.expand_dims(self.birdview_producer_obs.produce(agent_vehicle=self.vehicle)[:,:,0]*255, axis=-1)
        else:
            bev_obs = np.zeros((self.bev_obs_res[0], self.bev_obs_res[1], 1))

        if self.rgb_obs:
            while self.observation_buffer is None:
                pass
            rgb_obs = self.observation_buffer.copy()
            self.observation_buffer = None
        else:
            rgb_obs = np.zeros((self.rgb_obs_res[0], self.rgb_obs_res[1], 3))

        return {'rgb': rgb_obs, 'bev': bev_obs}

    def _get_viewer_image(self):
        if self.birdview_producer_display:
            bev_image = self.birdview_producer_display.produce(agent_vehicle=self.vehicle)[0]*255
        else:
            bev_image = np.zeros((self.bev_viewer_res[0], self.bev_viewer_res[1]))

        if self.rgb_display:
            while self.viewer_image_buffer is None:
                pass
            rgb_image = self.viewer_image_buffer.copy()
            self.viewer_image_buffer = None
        else:
            rgb_image = np.zeros((self.rgb_viewer_res[0], self.rgb_viewer_res[1]))

        return {'rgb': rgb_image, 'bev': bev_image}

    def _on_collision(self, event):
        if get_actor_display_name(event.other_actor) != "Road" and get_actor_display_name(event.other_actor) != "Roadline":
            self.collision = True
            if self.hud:
                self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

    def _set_rgb_observation_image(self, image):
        self.observation_buffer = image

    def _set_rgb_viewer_image(self, image):
        self.viewer_image_buffer = image



