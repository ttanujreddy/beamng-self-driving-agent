from beamngpy import BeamNGpy, Vehicle, Scenario
from beamngpy.sensors import Electrics, RoadsSensor, State
import csv, time, os

# Creates BeamNGy object that connects to beamng.
connection_object = BeamNGpy(host='localhost', port=64256, home="F:\BeamNG.tech.v0.38.3.0")

# Creates Scenario object that loads Hirochi Raceway
scenario = Scenario(level="hirochi_raceway", name="test1")

car = Vehicle("car1", model="sbr")

# Creates sensor object "electrics" and adds it to car
electrics = Electrics()
car.sensors.attach("electrics", electrics)

# Create state sensor
state = State()
car.sensors.attac("state", state)

scenario.add_vehicle(car, pos=(-408.48, 260.23, 25.14), rot_quat=(0, 0, -0.2799, 0.9600))

with connection_object.open() as bng:
    try:
        bng.scenario.stop()
    except:
        pass
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    roads = RoadsSensor("roads", bng, car, physics_update_time=1.00, is_visualised=True)
    time.sleep(5)

    while True:
        # TODO: Pull all neccessary data from sensors, prepare it and send to csv
        car.sensors.poll()

        speed = electrics.data.get("wheelspeed",0)
        steering = electrics.data.get("steering", 0)
        throttle = electrics.data.get("throttle", 0)
        brake = electrics.data.get("brake", 0)
        roads_data = roads.poll()
        print(roads_data) # Currently printing the entire roads sensor data dictionary
        # TODO: Isolate the necessary data from roads_data

        # TODO: Pull input data from the model and apply it to the vehicle

        time.sleep(1)