"""
Author(s): Nathan Fermo
Class: CS450-01
Date: 04/29/26
"""

from beamngpy import Vehicle


# Creates map and scenario names for this project setup.
TRACK_LEVEL = "hirochi_raceway"
TRACK_SCENARIO_NAME = "cs450_environment_test"

# Sets car on the racetrack (start of lap).
EGO_POS = (-408.48, 260.23, 25.14)
EGO_ROT_QUAT = (0.0, 0.0, -0.2799, 0.9600)

# Optional obstacle placements for environment testing.
# Placed right next to the ego car so it is easy to visually confirm.
OBSTACLE_1_POS = (-405.50, 260.23, 25.14)
OBSTACLE_1_ROT_QUAT = (0.0, 0.0, -0.2799, 0.9600)

OBSTACLE_2_POS = (-386.00, 257.00, 25.14)
OBSTACLE_2_ROT_QUAT = (0.0, 0.0, -0.2799, 0.9600)


def apply_environment_setup(scenario, ego_car, with_obstacles=False):
    # Adds the main car to the track at the lap start.
    scenario.add_vehicle(ego_car, pos=EGO_POS, rot_quat=EGO_ROT_QUAT)

    # Adds optional obstacle cars if requested.
    if with_obstacles:
        obstacle_1 = Vehicle("obstacle_1", model="etk800", color="Red")
        scenario.add_vehicle(
            obstacle_1,
            pos=OBSTACLE_1_POS,
            rot_quat=OBSTACLE_1_ROT_QUAT,
        )

        obstacle_2 = Vehicle("obstacle_2", model="etk800", color="Blue")
        scenario.add_vehicle(
            obstacle_2,
            pos=OBSTACLE_2_POS,
            rot_quat=OBSTACLE_2_ROT_QUAT,
        )
