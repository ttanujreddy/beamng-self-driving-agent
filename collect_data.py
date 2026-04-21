from beamngpy import BeamNGpy, Vehicle, Scenario
from beamngpy.sensors import Electrics, Camera
from PIL import Image
import csv, time, os

# TODO: replace with actual BeamNG install path

# Creates folder to store frames from camera
os.makedirs("frames", exist_ok=True)

# Creates BeamNGy object that connects to beamng.
connection_object = BeamNGpy(host='localhost', port=64256, home="insert file path")

# Creates Scenario object that loads Hirochi Raceway
scenario = Scenario(level="hirochi_raceway", name="test1")

car = Vehicle("car1", model="sbr")

# Creates sensor object "electrics" and adds it to car
electrics = Electrics()
car.sensors.attach("electrics", electrics)

# Sets car on the racetrack
scenario.add_vehicle(car, pos=(0,0,0), rot_quat=(0, 0, 0, 1))


with connection_object.open() as bng:
    bng.scenario.load(scenario)
    bng.scenario.start()

    camera = Camera("camera", bng, car)
    car.sensors.attach("camera", camera)


    with open('raw_data.csv', 'w', newline='') as f:
        
        # Creates raw_data.csv and writes column headers
        writer = csv.writer(f)
        writer.writerow(["Frame","Speed", "Steering", "Throttle", "Brake"])

        # Loop pulls data 10 times per second and writes it to the raw csv file.
        for i in range(500):
            car.sensors.poll()
            frame_filename = f"frames/frame_{i:04d}.png"
            camera_data = camera.poll()
            image = Image.fromarray(camera_data['colour'])
            image.save(frame_filename)
            speed = electrics.data.get("wheelspeed",0)
            steering = electrics.data.get("steering", 0)
            throttle = electrics.data.get("throttle", 0)
            brake = electrics.data.get("brake", 0)
            writer.writerow([frame_filename, speed, steering, throttle, brake])

            time.sleep(0.1)

