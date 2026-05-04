"""rl_reward.py - Shared reward function for RL training and demo evaluation.

Author(s): Tanuj Reddy Thummala
Class: CS450-01
Date: 04/29/26

This module defines one reward function for the whole RL branch.  rl_agent.py
uses it during REINFORCE training, and demo_rl_reward.py uses it without BeamNG
for a quick demonstration.

State format:
    [distFromCenter, headingAngle, curvature, roadWidth, drivability, Speed]

Action format:
    [Steering, Throttle, Brake]

Reward design:

Positive behavior:
    - forward speed aligned with road heading

Penalties:
    - leaving the road
    - being on a non-drivable surface
    - getting stuck
    - drifting away from the lane center
    - driving too fast in high curvature
    - pressing throttle and brake together
    - large steering magnitude

The exact weights are intentionally simple and explainable for presentation.
They can be tuned later without changing rl_agent.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from state_schema import safe_float


@dataclass
class RewardConfig:
    """Configuration values controlling reward shaping.

    Attributes:
        offroad_penalty: Reward returned when the car leaves the road.
        non_drivable_penalty: Reward returned when drivability indicates bad road.
        stuck_penalty: Reward returned when the car is almost stationary.
        low_speed_threshold: Speed below which the car is considered stuck.
        center_penalty_weight: Multiplier for distance from centerline.
        curvature_speed_penalty_weight: Penalty for speed on sharp turns.
        throttle_brake_conflict_weight: Penalty for throttle and brake together.
        steering_smoothness_weight: Small penalty for large steering magnitude.
    """

    offroad_penalty: float = -10.0
    non_drivable_penalty: float = -10.0
    stuck_penalty: float = -1.0
    low_speed_threshold: float = 0.5
    center_penalty_weight: float = 0.5
    curvature_speed_penalty_weight: float = 0.25
    throttle_brake_conflict_weight: float = 0.5
    steering_smoothness_weight: float = 0.05


DEFAULT_REWARD_CONFIG = RewardConfig()


def compute_reward(
    state: Iterable[float],
    action: Optional[Iterable[float]] = None,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> Tuple[float, bool]:
    """Compute a scalar reward and terminal flag from one driving state.

    Args:
        state: Iterable with six values in canonical state order:
            [distFromCenter, headingAngle, curvature, roadWidth, drivability, Speed]
        action: Optional iterable with three values:
            [Steering, Throttle, Brake]
            Passing action enables small smoothness/control-conflict penalties.
        config: Reward weights and thresholds.

    Returns:
        Tuple (reward, done):
            reward is the scalar signal used by RL.
            done is True when the episode should terminate.

    Raises:
        ValueError: If state does not contain exactly six values.
    """
    values = list(state)
    if len(values) != 6:
        raise ValueError(f"Expected 6 state values, got {len(values)}: {values}")

    dist_from_center, heading_angle, curvature, road_width, drivability, speed = [
        safe_float(value) for value in values
    ]

    half_width = road_width / 2.0 if road_width > 0 else 0.0

    # Terminal condition 1: clearly off-road.  distFromCenter is measured from
    # the lane centerline, so the road edge is roughly roadWidth / 2.
    if half_width > 0 and abs(dist_from_center) > half_width:
        return config.offroad_penalty, True

    # Terminal condition 2: BeamNG reports a non-drivable surface.
    if drivability <= 0:
        return config.non_drivable_penalty, True

    # Stuck penalty.  This discourages policies that simply brake forever, but
    # it does not terminate the episode because the car might recover.
    if speed < config.low_speed_threshold:
        return config.stuck_penalty, False

    # Forward progress reward.  cos(heading_angle) is near 1 when the vehicle is
    # aligned with the road, near 0 when perpendicular, and negative if going the
    # wrong way.  Multiplying by speed rewards useful forward motion.
    forward_reward = speed * math.cos(heading_angle)

    # Centerline penalty.  At the road edge center_ratio is about 1.0.
    center_ratio = abs(dist_from_center) / half_width if half_width > 0 else 0.0
    center_penalty = config.center_penalty_weight * center_ratio

    # Curvature-speed penalty.  This discourages high speed in sharper turns.
    curvature_speed_penalty = config.curvature_speed_penalty_weight * abs(curvature) * max(speed, 0.0)

    # Optional action-quality penalties.
    action_penalty = 0.0
    if action is not None:
        action_values = list(action)
        if len(action_values) == 3:
            steering, throttle, brake = [safe_float(value) for value in action_values]
            action_penalty += config.throttle_brake_conflict_weight * min(abs(throttle), abs(brake))
            action_penalty += config.steering_smoothness_weight * abs(steering)

    reward = forward_reward - center_penalty - curvature_speed_penalty - action_penalty
    return float(reward), False
