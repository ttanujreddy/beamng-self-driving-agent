""" agent.py - Run the simulation of the model specified
in the config file. This file is the final step in our
driving model process.

Author(s): Gregory Larson
Class: CS450-01
Date: 04/29/26
"""

import json

import model
from environment_task_setup import apply_environment_setup
from beamngpy import BeamNGpy, Vehicle, Scenario
from beamngpy.sensors import Electrics, RoadsSensor
from torch import Tensor

from state_schema import raw_sensors_to_state

class Agent():
    """ Agent class holds the model instance, as well as functions to
    communicate with the model. Requires an instance of
    beamngpy to recieve parameters.
    """
    def __init__(self):
        self.model = model.DrivingModel()

    def load_model(self, path="best_model.pth"):
        """ Loads the model specified into the model.
        By default, the model will load the model that was previously
        created by training_loop.ipynb TODO: check final filename
        """
        self.model.load(path)

    def get_state(self, vehicle: Vehicle, roads: RoadsSensor):
        """Get the vehicle state in the canonical model feature order.

        Returns:
            A list with six values in this order:
            [distFromCenter, headingAngle, curvature, roadWidth, drivability, Speed]
        """

        # Poll the given parameters
        try:
            vehicle.poll_sensors()
        except Exception as exc:
            raise Exception("Electrics sensor poll error") from exc
        
        try:
            raw_roads_data = roads.poll()
        except Exception as exc:
            raise Exception("Road sensor poll error") from exc

        # Extract data from the sensors, return as a list
        state = raw_sensors_to_state(raw_roads_data, vehicle)
        if state is None:
            raise Exception("No usable roads data given, please skip")
        
        return state

    def get_action(self, state):
        """ Get an action list from the model.
        The Tensor returned by the model should have values already
        within the needed bounds for the vehicle's controls.

        Returns: a list with 3 values in this order:
        0 - steering;
        1 - throttle;
        2 - brake
        """

        action = [0, 0, 0] # Initialize action list
        state_tensor = Tensor(state) # Convert state to tensor
        action_tensor = self.model.forward(state_tensor) # Get prediction from model
        action = action_tensor.tolist() # Convert

        return action

def update(agent: Agent, beamng: BeamNGpy, vehicle: Vehicle, roads: RoadsSensor):
    """ Step the game forward, pull sensor data, get an action list from
    the model, and send it to the vehicle.
    """
    beamng.control.step(10)

    # Get current state
    try:
        state = agent.get_state(vehicle, roads)
        print(state)
    except Exception as e:
        print(f"Error: {e}, skipping update")
        return 0

    # Get next move from model
    action = agent.get_action(state)
    action = action[0] # Reduce list to single dimension list

    # Execute next move
    try:
        vehicle.control(action[0], action[1], action[2])
    except Exception as e:
        print(f"Error: {e}")

    # Loop

def initialize():
    """ Initializes BeamNG.tech according to config.json.
    """

    # Use format config[key][key] to pull data into the script
    config = json.load(open("config.json", "r"))

    # Start up the Agent and load the specified model
    agent = Agent()
    agent.load_model(config["model-path"])

    # Initialize the game (will throw an error if game can't be found)
    print("Initializing BeamNG.tech...")
    beamng = BeamNGpy(host = config["beamng-host"], port = config["beamng-port"], home = config["beamng-path"])
    beamng.open()

    # Initialize the scenario, vehicle, and the electrics sensor
    scenario = Scenario(level = config["scenario-level"], name = config["scenario-name"])

    vehicle = Vehicle(config["vehicle-name"], model = config["vehicle-model"])

    electrics = Electrics()
    vehicle.sensors.attach("electrics", electrics)

    # Call function to spwan vehicles, no obstacles
    apply_environment_setup(scenario, vehicle)

    # Call functions to start the game, apply settings, and load the map
    scenario.make(beamng)
    beamng.settings.set_deterministic(60) # Turns off variance in physics, limits framerate
    beamng.control.pause()
    beamng.scenario.load(scenario)
    beamng.scenario.start()

    # Initialize roads sensor and attach to vehicle
    roads = RoadsSensor("roads", beamng, vehicle, physics_update_time = 0.1)
    beamng.control.step(180) # Wait until the sensors have fully initialized

    # Drive forward to engage the roads sensor
    for i in range(6):
        vehicle.control(0, 0.5, 0)
        beamng.control.step(10)

    return beamng, vehicle, roads, agent

if __name__ == "__main__":
    """ On running this file, start up the model and BeamNG.tech,
    then use the model to drive indefinitely.
    """

    # Initialization
    print("Running agent.py directly...")
    beamng, vehicle, roads, agent = initialize()
    
    # Loop until user closes the game
    while True:
        update(agent, beamng, vehicle, roads)
