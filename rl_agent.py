"""rl_agent.py - Experimental RL fine-tuning using REINFORCE.

Author(s): Brendel Zuniga, Tanuj Reddy Thummala
Class: CS450-01
Date: 04/29/26

THIS IS EXPERIMENTAL. REINFORCE is a high-variance Monte Carlo policy-gradient
algorithm, so best_model_rl.pth should be treated as a candidate to evaluate
against best_model.pth, not a guaranteed upgrade.

This file performs experimental reinforcement-learning fine-tuning on top of
the behavior-cloning model trained by training_loop.ipynb.

Main supervised pipeline:
    collect_data.py -> raw_data.csv
    process_data.py -> processed_data.csv
    training_loop.ipynb -> best_model.pth

Experimental RL extension:
    best_model.pth -> rl_agent.py -> best_model_rl.pth

Why RL after behavior cloning?
    The behavior-cloning model imitates actions from BeamNG AI or human driving
    records.  That is useful, but it only trains on states seen in the dataset.  If
    the model drifts into an unfamiliar state, it may not know how to recover.  RL
    fine-tuning attempts to improve the policy by letting it act in BeamNG and learn
    from reward feedback.

Algorithm:
This file implements a simple REINFORCE-style Monte Carlo policy-gradient loop:

    1. Load DrivingModel from best_model.pth.
    2. Wrap it as a stochastic Gaussian policy.
    3. Run one BeamNG episode.
    4. Store log probabilities of sampled actions.
    5. Compute rewards from rl_reward.py.
    6. Compute discounted returns.
    7. Update policy parameters with loss = -log_prob * return.

IMPORTANT limitation:
REINFORCE has high variance and may not improve within a small number of
episodes. Treat best_model_rl.pth as a candidate to compare against
best_model.pth, not as a guaranteed upgrade.

Usage:
    python rl_agent.py
    
Optional command-line arguments:
    --config config.json
    --beamng-home "(LOCATION OF BEAM.NG)"

Output:
    best_model_rl.pth
"""

from __future__ import annotations

import argparse
import json
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, RoadsSensor

from environment_task_setup import EGO_POS, EGO_ROT_QUAT
from environment_task_setup import apply_environment_setup
from model import DrivingModel
from rl_reward import compute_reward
from state_schema import raw_sensors_to_state

# -----------------------------------------------------------------------------
# RL hyperparameters
# -----------------------------------------------------------------------------
# Discount factor.  0.99 makes the agent care strongly about future reward.
GAMMA = 0.99

# Small learning rate so RL updates do not immediately destroy the behavior-
# cloning policy learned from supervised training.
LR = 1e-4

# Maximum number of model actions per episode.  Each step below advances BeamNG
# by 10 physics ticks.
MAX_STEPS = 150

# Number of full episodes to attempt.  Increase this only when BeamNG runs are
# stable and you have time to train.
NUM_EPISODES = 50

# Initial log standard deviation for action exploration.  exp(-1.0) is about
# 0.37, giving moderate exploration around the behavior-cloning action mean.
LOG_STD_INIT = -2.5

# BeamNG physics steps to wait after scenario start before using sensors.
WARMUP_STEPS = 180

# File paths for warm-start and RL output.
PRETRAINED_PATH = "best_model.pth"
SAVE_PATH = "best_model_rl.pth"
CONFIG_PATH = "config.json"

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for rl_agent.py.

    --beamng-home matches collect_data.py usage and lets each teammate run
    the RL script without editing config.json.
    """
    parser = argparse.ArgumentParser(
        description="Experimental REINFORCE fine-tuning for BeamNG self-driving agent."
    )

    parser.add_argument(
        "--beamng-home",
        default=None,
        help="Optional BeamNG.tech install folder. Overrides beamng-path in config.json.",
    )

    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="Path to config.json. Defaults to config.json.",
    )

    return parser.parse_args()

class StochasticPolicy(nn.Module):
    """Wrap DrivingModel as a stochastic Gaussian policy for REINFORCE.

    DrivingModel normally outputs one deterministic action.  REINFORCE needs a
    stochastic policy so it can explore and compute log_prob(action).  This
    wrapper treats DrivingModel output as the mean of a Gaussian distribution and
    learns one log standard deviation per action dimension.

    Action dimensions:
        0. Steering in [-1, 1]
        1. Throttle in [0, 1]
        2. Brake in [0, 1]
    """

    def __init__(self, pretrained_path: str | None = None):
        """Create the stochastic policy and optionally load pretrained weights.

        Args:
            pretrained_path: Optional .pth checkpoint from behavior cloning.
        """
        super().__init__()
        self.base = DrivingModel()

        if pretrained_path:
            try:
                self.base.load(pretrained_path)
                print(f"Loaded pretrained weights from '{pretrained_path}'.")
            except FileNotFoundError:
                print(f"No pretrained weights found at '{pretrained_path}'. Training from scratch.")

        # Disable dropout inside DrivingModel.  RL exploration should come from
        # the Normal distribution below, not from random hidden-layer dropout.
        # eval() does not disable gradients; it only changes modules like Dropout.
        self.base.eval()

        # Learnable exploration parameter.  Using log_std instead of std keeps
        # std positive because std = exp(log_std).
        self.log_std = nn.Parameter(torch.full((3,), LOG_STD_INIT))

    def forward(self, x):
        """Return the deterministic mean action from the base DrivingModel."""
        return self.base(x)

    def get_action(self, state: List[float]) -> Tuple[List[float], torch.Tensor]:
        """Sample one action and return its log probability.

        Args:
            state: Six-value canonical state vector.

        Returns:
            Tuple (action, log_prob):
                action is [steering, throttle, brake] clipped to valid ranges.
                log_prob is a scalar tensor kept in the computation graph for
                the REINFORCE update.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # mean shape is (1, 3).  Gradients flow from log_prob back through mean
        # and into the base DrivingModel parameters.
        mean = self.forward(state_tensor)
        std = torch.exp(self.log_std).clamp(min=1e-4)
        dist = Normal(mean, std)

        # TODO/Future improvement:
        # The log probability is computed from the raw sampled action, while the action
        # sent to BeamNG is clipped to valid control ranges. This makes the REINFORCE
        # update an approximation because the logged action and executed action can differ.
        # A more correct implementation would use a squashed distribution with proper
        # log-prob correction, or compute the log probability for the exact bounded
        # action sent to the simulator.
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        action_tensor = raw_action.squeeze(0)
        steering = action_tensor[0].clamp(-1.0, 1.0).item()
        throttle = action_tensor[1].clamp(0.0, 1.0).item()
        brake = action_tensor[2].clamp(0.0, 1.0).item()

        return [steering, throttle, brake], log_prob.squeeze(0)

    def save_base_model(self, path: str) -> None:
        """Save only the base DrivingModel weights for use by agent.py.

        agent.py expects a normal DrivingModel checkpoint, not the full
        StochasticPolicy object.  This method saves self.base so the resulting
        file can be loaded exactly like best_model.pth.
        """
        self.base.save(path)

def compute_returns(rewards: List[float]) -> torch.Tensor:
    """Compute normalized discounted returns for one episode.

    For each timestep t:

        G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    Returns are normalized to reduce gradient variance.

    Args:
        rewards: Per-step rewards collected during an episode.

    Returns:
        torch.Tensor with one return value per reward.
    """
    G = 0.0
    returns = []

    for reward in reversed(rewards):
        G = reward + GAMMA * G
        returns.insert(0, G)

    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    if len(returns_tensor) > 1 and returns_tensor.std() > 1e-8:
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

    return returns_tensor

def get_state(vehicle: Vehicle, roads_sensor: RoadsSensor):
    """Poll BeamNG sensors and return (state, ok).

    This mirrors Agent.get_state() but returns a success flag instead of raising.
    RL training can skip bad frames without killing the whole training run.

    Args:
        vehicle: BeamNG Vehicle with Electrics sensor.
        roads_sensor: BeamNG RoadsSensor.

    Returns:
        Tuple (state, ok).  state is None when ok is False.
    """
    try:
        vehicle.poll_sensors()
    except Exception as exc:
        print(f"  Electrics poll error: {exc}")
        return None, False

    try:
        raw_roads_data = roads_sensor.poll()
    except Exception as exc:
        print(f"  Roads sensor poll error: {exc}")
        return None, False

    state = raw_sensors_to_state(raw_roads_data, vehicle)
    if state is None:
        return None, False

    return state, True

def reset_episode(beamng: BeamNGpy, vehicle: Vehicle) -> None:
    """Recover the vehicle and allow physics to settle before an episode.

    Args:
        beamng: Open BeamNGpy connection.
        vehicle: Vehicle being trained.
    """
    vehicle.teleport(pos=EGO_POS, rot_quat=EGO_ROT_QUAT)
    vehicle.control(steering=0, throttle=0, brake=1)
    beamng.control.step(60)

    vehicle.control(steering=0, throttle=0.4, brake=0)
    beamng.control.step(20)

def run_episode(beamng: BeamNGpy, vehicle: Vehicle, roads_sensor: RoadsSensor, policy: StochasticPolicy):
    """Run one episode and collect REINFORCE training data.

    Correct RL order:
        1. Observe state.
        2. Sample action and log_prob.
        3. Apply action to vehicle.
        4. Step simulator.
        5. Observe next_state.
        6. Compute reward from next_state and action.

    Args:
        beamng: Open BeamNGpy connection.
        vehicle: Vehicle controlled by the policy.
        roads_sensor: RoadsSensor attached to vehicle.
        policy: StochasticPolicy being trained.

    Returns:
        Tuple (log_probs, rewards, total_reward).
    """
    log_probs = []
    rewards = []
    total_reward = 0.0

    state, ok = get_state(vehicle, roads_sensor)
    if not ok:
        return log_probs, rewards, total_reward

    for step in range(MAX_STEPS):
        action, log_prob = policy.get_action(state)

        try:
            vehicle.control(steering=action[0], throttle=action[1], brake=action[2])
        except Exception as exc:
            print(f"  Vehicle control error: {exc}")
            beamng.control.step(10)
            state, ok = get_state(vehicle, roads_sensor)
            if not ok:
                break
            continue

        # Let the sampled action affect the simulator before scoring reward.
        beamng.control.step(10)

        next_state, ok = get_state(vehicle, roads_sensor)
        if not ok:
            break

        reward, done = compute_reward(next_state, action)

        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward

        state = next_state

        if done:
            print(f"  Terminal reward condition at step {step}.")
            break

    return log_probs, rewards, total_reward

def reinforce_update(policy: StochasticPolicy, optimizer: torch.optim.Optimizer, log_probs, rewards) -> float:
    """Apply one REINFORCE update from one episode.

    Loss:
        loss = -sum(log_prob(action_t) * discounted_return_t)

    Args:
        policy: StochasticPolicy being optimized.
        optimizer: PyTorch optimizer.
        log_probs: List of scalar log-probability tensors.
        rewards: List of float rewards from the same episode.

    Returns:
        Scalar loss value for logging.  Returns 0.0 when no update occurs.
    """
    if not log_probs or not rewards:
        return 0.0

    returns = compute_returns(rewards)
    log_probs_tensor = torch.stack(log_probs)

    loss = -(log_probs_tensor * returns).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())

def initialize_beamng(config_path: str = CONFIG_PATH, beamng_home: str | None = None):
    """Initialize BeamNG objects for RL training.

    Args:
        config_path: Path to config.json.
        beamng_home: Optional BeamNG.tech install folder. If provided, this
            overrides the "beamng-path" value from config.json.

    Returns:
        Tuple (beamng, vehicle, roads_sensor).
    """
    with open(config_path, "r") as file:
        config = json.load(file)

    # Allows command-line override:
    # python rl_agent.py --beamng-home "C:/BeamNG.tech.v0.38.5.0"
    if beamng_home is not None:
        config["beamng-path"] = beamng_home

    beamng = BeamNGpy(
        host=config["beamng-host"],
        port=config["beamng-port"],
        home=config["beamng-path"],
    )
    beamng.open()

    scenario = Scenario(level=config["scenario-level"], name=config["scenario-name"])
    vehicle = Vehicle(config["vehicle-name"], model=config["vehicle-model"])

    electrics = Electrics()
    vehicle.sensors.attach("electrics", electrics)

    apply_environment_setup(scenario, vehicle)

    scenario.make(beamng)
    beamng.settings.set_deterministic(60)
    beamng.control.pause()
    beamng.scenario.load(scenario)
    beamng.scenario.start()

    roads_sensor = RoadsSensor("roads", beamng, vehicle, physics_update_time=0.1)
    beamng.control.step(WARMUP_STEPS)

    # Move slightly so RoadsSensor starts returning useful lane information.
    for _ in range(6):
        vehicle.control(steering=0, throttle=0.5, brake=0)
        beamng.control.step(10)

    return beamng, vehicle, roads_sensor

def train_rl(config_path: str = CONFIG_PATH, beamng_home: str | None = None) -> None:
    """Run the full experimental RL fine-tuning process."""
    policy = StochasticPolicy(pretrained_path=PRETRAINED_PATH)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    beamng, vehicle, roads_sensor = initialize_beamng(
        config_path=config_path,
        beamng_home=beamng_home,
    )

    best_reward = float("-inf")

    try:
        for episode in range(1, NUM_EPISODES + 1):
            print(f"\nEpisode {episode}/{NUM_EPISODES}")

            reset_episode(beamng, vehicle)
            log_probs, rewards, total_reward = run_episode(beamng, vehicle, roads_sensor, policy)
            loss = reinforce_update(policy, optimizer, log_probs, rewards)

            print(
                f"Episode reward={total_reward:.3f} "
                f"steps={len(rewards)} loss={loss:.6f} "
                f"std={torch.exp(policy.log_std).detach().tolist()}"
            )

            if total_reward > best_reward and rewards:
                best_reward = total_reward
                policy.save_base_model(SAVE_PATH)
                print(f"  Saved new best RL model to {SAVE_PATH}")

    finally:
        try:
            beamng.disconnect()
        except Exception:
            pass

    print(f"Best episode reward: {best_reward:.3f}")

if __name__ == "__main__":
    args = parse_args()
    train_rl(config_path=args.config, beamng_home=args.beamng_home)
