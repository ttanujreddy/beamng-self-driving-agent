"""demo_rl_reward.py - BeamNG-free demonstration of the reward function.

Author(s): Tanuj Reddy Thummala
Class: CS450-01
Date: 04/29/26

Run this when BeamNG is not available but you still want to verify and explain
the reinforcement-learning reward design.

It is useful for:

    1. Verifying that rl_reward.py imports and runs.
    2. Demonstrating the RL reward design without the complexity of BeamNG.
    3. Sanity-checking that good states get better rewards than bad states.

Usage:
    python demo_rl_reward.py
"""

from rl_reward import compute_reward


# Each tuple is: (human-readable label, state, action).
# State order is documented in state_schema.py and rl_reward.py.
DEMO_CASES = [
    (
        "centered, aligned, moving",
        [0.0, 0.0, 0.001, 8.0, 1.0, 12.0],
        [0.0, 0.5, 0.0],
    ),
    (
        "near road edge",
        [3.2, 0.1, 0.001, 8.0, 1.0, 8.0],
        [0.2, 0.4, 0.0],
    ),
    (
        "too fast into curve",
        [0.5, 0.2, 0.08, 8.0, 1.0, 20.0],
        [0.4, 0.8, 0.0],
    ),
    (
        "stopped",
        [0.0, 0.0, 0.001, 8.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ),
    (
        "off road",
        [5.0, 0.0, 0.001, 8.0, 1.0, 10.0],
        [0.0, 0.5, 0.0],
    ),
]


def main() -> None:
    """Print reward values for several manually chosen driving situations."""
    print("RL reward demo")
    print("State = [distFromCenter, headingAngle, curvature, roadWidth, drivability, Speed]")
    print("Action = [Steering, Throttle, Brake]\n")

    for name, state, action in DEMO_CASES:
        reward, done = compute_reward(state, action)
        print(f"{name:24s} reward={reward:8.3f} done={done}")


if __name__ == "__main__":
    main()
