from beamngpy import BeamNGpy, Vehicle, Scenario
from beamngpy.sensors import Electrics
import csv, time

connection_object = BeamNGpy(host='localhost', port=64256, home="insert file path")

scenario = Scenario(level="hirochi_raceway", name="test1")

car = Vehicle("car1", model="sbr")

electrics = Electrics()
car.sensors.attach("electrics", electrics)

scenario.add_vehicle(car, pos=(0,0,0), rot_quat=(0, 0, 0, 1))

with connection_object.open() as bng:
    bng.scenario.load(scenario)
    bng.scenario.start()

    with open('filename.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Speed", "Steering", "Throttle", "Brake"])

        for i in range(500):
            car.sensors.poll()
            speed = electrics.data.get("wheelspeed",0)
            steering = electrics.data.get("steering", 0)
            throttle = electrics.data.get("throttle", 0)
            brake = electrics.data.get("brake", 0)
            writer.writerow([speed, steering, throttle, brake])

            time.sleep(0.1)



    
    # everything else goes inside here