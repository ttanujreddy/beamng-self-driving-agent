"""
Data collection script for the self-driving project.
Connects to BeamNG, drives the SBR car around Hirochi Raceway with the AI driver,
logs road sensor and electrics data to raw_data.csv for downstream training.
"""

from beamngpy import BeamNGpy, Vehicle, Scenario
from beamngpy.sensors import Electrics, RoadsSensor
import csv, time, argparse

from environment_task_setup import (
    TRACK_LEVEL,
    TRACK_SCENARIO_NAME,
    apply_environment_setup,
)

# Run commands (PowerShell/CMD): (replace the path with your own if it's different)
# python collect_data.py --beamng-home "C:/BeamNGTech/BeamNG.tech.v0.38.3.0" --frames 30 --with-obstacles --startup-delay 10 --warmup-polls 20 --keep-open

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beamng-home", required=True, help="BeamNG.tech folder path.")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to collect.")
    parser.add_argument("--with-obstacles", action="store_true", help="Enable obstacle cars.")
    parser.add_argument("--keep-open", action="store_true", help="Wait for Enter before disconnecting.")
    parser.add_argument("--startup-delay", type=float, default=8.0, help="Seconds to wait before first road sensor poll.")
    parser.add_argument("--warmup-polls", type=int, default=10, help="Number of warmup polls before capture loop.")
    parser.add_argument("--ai-mode", default="span", help="BeamNG AI mode: span, traffic, random, manual, disabled.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

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
        # Makes sure scenario is stopped before starting a new one
        try:
            bng.scenario.stop()
        except Exception:
            pass
        scenario.make(bng)
        bng.scenario.load(scenario)
        bng.scenario.start()

        time.sleep(args.startup_delay)  # let the simulator finish spinning up rendering/physics

        # Create sensor BEFORE AI setup so it registers cleanly with the vehicle.
        # Don't call car.connect(bng) — scenario.start() already handles vehicle connection,
        # and calling connect manually triggers a reconnect that orphans sensors.
        roads_sensor = RoadsSensor("roads1", bng, car, is_send_immediately=True, is_visualised=True)
        time.sleep(1)  # give sensor time to fully register on VE side

        # Sets up an AI driver to drive the track for training
        car.ai.set_mode(args.ai_mode)
        car.ai.set_aggression(0.7)
        car.ai.set_speed(15, mode="set")

        print("Scenario started. Beginning frame capture...")
        print(f"Obstacles enabled: {args.with_obstacles}")

        # Warms up the road sensor so first frames are less likely to fail.
        for i in range(args.warmup_polls):
            try:
                warmup_data = roads_sensor.poll()
                # TODO: check if data is valid once we know the dict structure
            except Exception:
                pass
            time.sleep(0.1)

        with open("raw_data.csv", "w", newline="") as f:
            # Creates raw_data.csv and writes column headers
            writer = csv.writer(f)
            writer.writerow(["distFromCenter","headingAngle","curvature","roadWidth","drivability", "Speed", "Steering", "Throttle", "Brake"])

            # Loop pulls data 10 times per second and writes it to the raw csv file.
            saved_frames = 0
            for i in range(args.frames):
                try:
                    car.sensors.poll()
                except Exception as e:
                    print(f"Sensor poll error at frame {i}: {e}")
                    time.sleep(0.2)
                    continue
                    

                # Pulls road sensor and stores it in "roads_data"
                try:
                    roads_data = roads_sensor.poll()
                except Exception as e:
                    print(f"Road Sensor poll error at {i}: {e}")
                    time.sleep(0.2)
                    continue

                
                if not isinstance(roads_data, dict) or not roads_data:
                    print(f"Skipping frame {i}, no road data yet")
                    time.sleep(0.1)
                    continue

                distFromCenter = roads_data.get("dist2CL", 0)
                headingAngle = roads_data.get("headingAngle", 0)
                radius = roads_data.get("roadRadius", 0)
                
                #  Handle straight roads (NaN) and missing data (0) — both mean curvature 0.
                if radius == 0 or radius != radius:
                    curvature = 0
                else:
                    curvature = 1 / radius

                # BeamNG returns half-width (center to edge); double it for full road width.
                roadWidth = 2 * roads_data.get("halfWidth", 0)
                drivability = roads_data.get("drivability", 0)

                Speed = electrics.data.get("wheelspeed", 0)
                Steering = electrics.data.get("steering_input", 0)
                Throttle = electrics.data.get("throttle", 0)
                Brake = electrics.data.get("brake", 0)
                writer.writerow([distFromCenter, headingAngle, curvature, roadWidth, drivability, Speed, Steering, Throttle, Brake])
                saved_frames += 1
                time.sleep(0.1)

            print(f"Capture finished. Saved {saved_frames}/{args.frames} frames.")
            print("Output file: raw_data.csv")

        if args.keep_open:
            input("Press Enter to close BeamNG connection...")
