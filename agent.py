import model
from environment_task_setup import apply_environment_setup
from beamngpy import BeamNGpy, Vehicle, Scenario
from beamngpy.sensors import Electrics, RoadsSensor
from torch import Tensor
import random
import json

    # Input format notes from Matt:
    # DistFromCenter — distance from lane centerline, scaled to [-1, 1]
    # HeadingAngle — car angle vs road direction, scaled to [-1, 1]
    # XCurvature, YCurvature — road curvature, scaled to [-1, 1]
    # RoadWidth — scaled to [0, 1]
    # Drivability — already 0-1 from BeamNG

    # Output format notes from Matt:
    # Steering ∈ [-1, 1] (raw, not normalized — 0 means straight)
    # Throttle ∈ [0, 1] (raw)
    # Brake ∈ [0, 1] (raw)
    # Speed ∈ [0, 1] (scaled from m/s)

class Agent():
    def __init__(self):
        self.model = model.DrivingModel()

        # The following declarations are for Reinforcement Learning
        self.n_attempts = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0 # Discount rate
        self.exploration = False # Enable exploration
        self.memory = None # TODO: Implement for RL
        self.trainer = None # TODO: Make a trainer class for reinforcement learning

    def get_state(self, roads: RoadsSensor):
        raw_roads_data = roads.poll()
        roads_data = raw_roads_data[0]
        state = [roads_data["dist2CL"], roads_data["headingAngle"], roads_data["roadRadius"], roads_data["halfWidth"], roads_data["drivability"]]
        return state

    def get_action(self, state):
        if self.exploration:
            self.epsilon = 80 - self.n_attempts
        else:
            self.epsilon = 0

        action = [0, 0, 0] # Initialize action list

        if random.randint(0, 200) < self.epsilon: # Chance to explore
            action[0] = float(random.randint(-100, 100)) / 100
            action[1] = float(random.randint(0, 100)) / 100
            action[2] = float(random.randint(0, 100)) / 100
        else:
            state_tensor = Tensor(state) # Convert state to tensor
            action_tensor = self.model.forward(state_tensor) # Get prediction from model
            action = action_tensor.tolist() # Convert

        return action
    
    def remember(self):
        # TODO: Implement for RL
        pass

    def train_long_memory(self):
        # TODO: Implement for RL
        pass

    def train_short_memory(self):
        # TODO: Implement for RL
        pass

def update(agent: Agent, beamng: BeamNGpy, scenario: Scenario, vehicle: Vehicle, roads: RoadsSensor, electrics: Electrics):
    beamng.control.step(10)

    # Get current state
    state = agent.get_state(roads)
    print(state)

    # Get next move from model
    action = agent.get_action(state)

    # Execute next move
    vehicle.control(action[0], action[1], action[2])
    # Loop

def initialize():
    # Use format config[key][key] to pull data into the script
    config = json.load(open("config.json", "r"))

    print("Initializing BeamNG.tech...")

    # TODO: Consider replacing this whole block with an initialization function
    beamng = BeamNGpy(host = config["beamng-host"], port = config["beamng-port"], home = config["beamng-path"])
    beamng.open()

    scenario = Scenario(level = config["scenario-level"], name = config["scenario-name"])

    vehicle = Vehicle(config["vehicle-name"], model = config["vehicle-model"])

    electrics = Electrics()

    vehicle.sensors.attach("electrics", electrics)

    apply_environment_setup(scenario, vehicle)

    scenario.make(beamng)
    beamng.settings.set_deterministic(60)
    beamng.control.pause()
    beamng.scenario.load(scenario)
    beamng.scenario.start()
    # Block end

    # Initialize roads sensor and attach to vehicle
    roads = RoadsSensor("roads", beamng, vehicle, physics_update_time = 0.1)
    beamng.control.step(180)

    return beamng, scenario, vehicle, electrics, roads

if __name__ == "__main__":
    print("Running agent.py directly.")
    beamng, scenario, vehicle, electrics, roads = initialize()
    agent = Agent()
    
    # Begin loops
    while True:
        update(agent, beamng, scenario, vehicle, roads, electrics)
