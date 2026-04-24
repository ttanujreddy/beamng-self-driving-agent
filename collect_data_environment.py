from beamngpy import BeamNGpy, Vehicle, Scenario
from beamngpy.sensors import Electrics, Camera
from PIL import Image
import csv, time, os, argparse

from environment_task_setup import (
    TRACK_LEVEL,
    TRACK_SCENARIO_NAME,
    apply_environment_setup,
)

# Run commands (PowerShell/CMD): (replace the path with your own if it's different)
# python collect_data_environment.py --beamng-home "C:/BeamNGTech/BeamNG.tech.v0.38.3.0" --frames 30 --with-obstacles --startup-delay 10 --warmup-polls 20 --keep-open

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beamng-home", required=True, help="BeamNG.tech folder path.")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to collect.")
    parser.add_argument("--with-obstacles", action="store_true", help="Enable obstacle cars.")
    parser.add_argument("--keep-open", action="store_true", help="Wait for Enter before disconnecting.")
    parser.add_argument("--startup-delay", type=float, default=8.0, help="Seconds to wait before first camera poll.")
    parser.add_argument("--warmup-polls", type=int, default=10, help="Number of warmup polls before capture loop.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Creates folder to store frames from camera
    os.makedirs("frames", exist_ok=True)

    # Creates BeamNGy object that connects to beamng.
    connection_object = BeamNGpy(host="localhost", port=64256, home=args.beamng_home)

    # Creates Scenario object that loads Hirochi Raceway
    scenario = Scenario(level=TRACK_LEVEL, name=TRACK_SCENARIO_NAME)

    car = Vehicle("car1", model="sbr")

    # Creates sensor object "electrics" and adds it to car
    electrics = Electrics()
    car.sensors.attach("electrics", electrics)

    # Sets car and optional obstacles on the racetrack
    apply_environment_setup(scenario, car, with_obstacles=args.with_obstacles)

    with connection_object.open() as bng:
        try:
            bng.scenario.stop()
        except:
            pass
        scenario.make(bng)
        bng.scenario.load(scenario)
        bng.scenario.start()
        time.sleep(args.startup_delay)  # wait for camera to initialize

        camera = Camera("camera", bng, car)
        print("Scenario started. Beginning frame capture...")
        print(f"Obstacles enabled: {args.with_obstacles}")

        # Warms up the camera so first frames are less likely to fail.
        for i in range(args.warmup_polls):
            try:
                warmup_data = camera.poll()
                if warmup_data["colour"] is not None:
                    break
            except Exception:
                pass
            time.sleep(0.1)

        with open("raw_data_env_test.csv", "w", newline="") as f:
            # Creates raw_data_env_test.csv and writes column headers
            writer = csv.writer(f)
            writer.writerow(["Frame", "Speed", "Steering", "Throttle", "Brake"])

            # Loop pulls data 10 times per second and writes it to the raw csv file.
            saved_frames = 0
            for i in range(args.frames):
                try:
                    car.sensors.poll()
                except Exception as e:
                    print(f"Sensor poll error at frame {i}: {e}")
                    time.sleep(0.2)
                    continue

                # Creates a unique file name for image frames
                frame_filename = f"frames/frame_{i:04d}.png"

                # Pulls image frame and stores it in "camera_data"
                try:
                    camera_data = camera.poll()
                except Exception as e:
                    print(f"Camera poll error at frame {i}: {e}")
                    time.sleep(0.2)
                    continue

                # Checks to see if there is a frame to save.
                if camera_data["colour"] is None:
                    print(f"No camera colour frame at index {i}")
                    continue

                camera_data["colour"].save(frame_filename)
                saved_frames += 1

                speed = electrics.data.get("wheelspeed", 0)
                steering = electrics.data.get("steering", 0)
                throttle = electrics.data.get("throttle", 0)
                brake = electrics.data.get("brake", 0)
                writer.writerow([frame_filename, speed, steering, throttle, brake])

                time.sleep(0.1)

            print(f"Capture finished. Saved {saved_frames}/{args.frames} frames.")
            print("Output file: raw_data_env_test.csv")

        if args.keep_open:
            input("Press Enter to close BeamNG connection...")
