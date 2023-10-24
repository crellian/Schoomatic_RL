#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""
import carla
import logging
import random
import psutil

def is_used(port):
    """Checks whether or not a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]

class NPCClass:
    def __init__(self, client):
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.client = client
        self.world = self.client.get_world()

    def _spawn_npcs(self, n_vehicles, n_obs_vehicles, n_walkers, n_obs_walkers):
        """Spawns vehicles and walkers, also setting up the Traffic Manager and its parameters"""
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # Spawn vehicles
        spawn_points = self.world.get_map().get_spawn_points()
        n_spawn_points = len(spawn_points)

        if n_vehicles + n_obs_vehicles < n_spawn_points:
            random.shuffle(spawn_points)
        elif n_vehicles + n_obs_vehicles > n_spawn_points:
            logging.warning("{} vehicles were requested, but there were only {} available spawn points"
                            .format(n_vehicles, n_spawn_points))
            n_vehicles = n_spawn_points / 2
            n_obs_vehicles = n_obs_vehicles / 2

        v_batch = []
        v_blueprints = self.world.get_blueprint_library().filter("vehicle.*")

        for n, transform in enumerate(spawn_points):
            if n >= n_vehicles + n_obs_vehicles:
                break
            v_blueprint = random.choice(v_blueprints)
            if v_blueprint.has_attribute('color'):
                color = random.choice(v_blueprint.get_attribute('color').recommended_values)
                v_blueprint.set_attribute('color', color)
            v_blueprint.set_attribute('role_name', 'autopilot')

            transform.location.z += 1

            tm_port = random.randint(2000, 3000)
            while is_used(tm_port):
                print("Traffic manager's port " + str(tm_port) + " is already being used. Checking the next one")
                tm_port += 1

            if n < n_vehicles:
                v_batch.append(SpawnActor(v_blueprint, transform)
                               .then(SetAutopilot(FutureActor, True, tm_port)))
            else:
                v_batch.append(SpawnActor(v_blueprint, transform))

        results = self.client.apply_batch_sync(v_batch)
        if len(results) < n_vehicles + n_obs_vehicles:
            logging.warning("{} vehicles were requested but could only spawn {}"
                            .format(n_vehicles + n_obs_vehicles, len(results)))
        vehicles_id_list = [r.actor_id for r in results if not r.error]

        # Spawn the walkers
        spawn_locations = [self.world.get_random_location_from_navigation() for i in range(n_walkers + n_obs_walkers)]

        w_batch = []
        w_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")

        for spawn_location in spawn_locations:
            w_blueprint = random.choice(w_blueprints)
            if w_blueprint.has_attribute('is_invincible'):
                w_blueprint.set_attribute('is_invincible', 'false')
            w_batch.append(SpawnActor(w_blueprint, carla.Transform(spawn_location)))

        results = self.client.apply_batch_sync(w_batch)
        if len(results) < n_walkers + n_obs_walkers:
            logging.warning("Could only spawn {} out of the {} requested walkers."
                            .format(len(results), n_walkers + n_obs_walkers))
        walkers_id_list = [r.actor_id for r in results if not r.error]

        # Spawn the walker controllers
        wc_batch = []
        wc_blueprint = self.world.get_blueprint_library().find('controller.ai.walker')

        random.shuffle(walkers_id_list)
        for walker_id in walkers_id_list[:n_walkers]:
            wc_batch.append(SpawnActor(wc_blueprint, carla.Transform(), walker_id))

        results = self.client.apply_batch_sync(wc_batch)
        if len(results) < n_walkers:
            logging.warning("Only {} out of {} controllers could be created. Some walkers might be stopped"
                            .format(len(results), n_walkers))
        controllers_id_list = [r.actor_id for r in results if not r.error]

        self.world.tick()

        for controller in self.world.get_actors(controllers_id_list):
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())

        self.world.tick()
        self.actors = self.world.get_actors(vehicles_id_list + walkers_id_list + controllers_id_list)

        return self.actors

    def remove_npcs(self):
        print('Destroying %d NPC vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        # stop walker controllers (list is [controler, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('Destroying %d NPC walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])