import dummynet     # TODO: replace with final model file "model"
from environment_task_setup import apply_environment_setup
from beamngpy import BeamNGpy, Vehicle, Scenario
from beamngpy.sensors import Electrics, RoadsSensor
import json
import csv, time, os

def run_cycle(bng: BeamNGpy, vehicle: Vehicle, rs: RoadsSensor):
    bng.control.step(10)

    roads_data = rs.poll()

    # TODO: Figure out what format to send the data in
    # Input format notes from Matt:
    # DistFromCenter — distance from lane centerline, scaled to [-1, 1]
    # HeadingAngle — car angle vs road direction, scaled to [-1, 1]
    # XCurvature, YCurvature — road curvature, scaled to [-1, 1]
    # RoadWidth — scaled to [0, 1]
    # Drivability — already 0-1 from BeamNG

    # Send data to model, recieve output here
    ## output = model.forward(roads_data)
    output = dummynet.predict(roads_data)

    # TODO: Figure out how the data is being recieved
    # Output format notes from Matt:
    # Steering ∈ [-1, 1] (raw, not normalized — 0 means straight)
    # Throttle ∈ [0, 1] (raw)
    # Brake ∈ [0, 1] (raw)
    # Speed ∈ [0, 1] (scaled from m/s)

    vehicle.control(output[0], output[1], output[2])

    # Resume

if __name__ == "__main__":
    print("Running agent.py directly.")

    # Use format config[key][key] to pull data into the script
    config = json.load(open("config.json", "r"))

    print("Initializing BeamNG.tech...")

    # TODO: Consider replacing this whole block with an initialization function
    bng = BeamNGpy(host = config["beamng-host"], port = config["beamng-host"], home = config["beamng-path"])

    scenario = Scenario(level = config["scenario-level"], name = config["scenario-name"])

    vehicle = Vehicle(config["vehicle-name"], model = config["vehicle-model"])

    electrics = Electrics()

    vehicle.sensors.attach("electrics", electrics)

    apply_environment_setup(scenario, vehicle)

    scenario.make(bng)
    bng.set_deterministic(60)
    bng.control.pause()
    bng.scenario.load(scenario)
    bng.scenario.start()
    # Block end

    # Initialize roads sensor and attach to vehicle
    roads = RoadsSensor("roads", bng, config["vehicle-name"])

    while(True):
        run_cycle()



    