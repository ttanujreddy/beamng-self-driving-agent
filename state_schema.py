"""state_schema.py - Shared driving data schema and sensor conversion helpers.

Author(s): Matt Gallenberger, Brendel Zuniga, Gregory Larson, Tanuj Reddy Thummala
Class: CS450-01
Date: 04/29/26

This file is the single source of truth for the driving model inputs and outputs.

The training pipeline and live agents must agree on the exact same feature order:
    X = [distFromCenter, headingAngle, curvature, roadWidth, drivability, Speed]
    y = [Steering, Throttle, Brake]

BeamNG RoadsSensor returns roadRadius and halfWidth, but collect_data.py stores
curvature and roadWidth:
    curvature = 1 / roadRadius, except straight/missing roads -> 0
    roadWidth = 2 * halfWidth

agent.py and rl_agent.py should use this module instead of re-implementing those
transforms inline.  That keeps training, inference, and RL fine-tuning consistent.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional


# -----------------------------------------------------------------------------
# Canonical dataset columns
# -----------------------------------------------------------------------------
# These names must match collect_data.py, process_data.py, utils.py, and the
# training notebook.  The model architecture in model.py assumes exactly six
# input values in this order.
FEATURE_COLUMNS = [
    "distFromCenter",
    "headingAngle",
    "curvature",
    "roadWidth",
    "drivability",
    "Speed",
]

# These are the three continuous control values learned by the behavior cloning
# model and emitted by the policy during live driving.
TARGET_COLUMNS = [
    "Steering",
    "Throttle",
    "Brake",
]

# A valid training row must contain all model inputs and all target controls.
REQUIRED_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS

# Targets are clipped rather than min-max normalized.  Steering must remain
# signed because steering=0 means straight, negative means one direction, and
# positive means the other direction.  Throttle/brake already use [0, 1].
TARGET_CLIP_RANGES = {
    "Steering": (-1.0, 1.0),
    "Throttle": (0.0, 1.0),
    "Brake": (0.0, 1.0),
}

# RoadsSensor keys used to recognize whether a dictionary contains road data.
ROAD_KEYS = {"dist2CL", "headingAngle", "roadRadius", "halfWidth", "drivability"}


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a simulator/CSV value into a finite float.

    BeamNG sensor values can occasionally be missing, None, NaN, or infinity.
    Neural networks and reward functions should not receive those values.

    Args:
        value: Object to convert to float.
        default: Value returned when conversion fails or result is non-finite.

    Returns:
        A finite float.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default

    return result


def clamp(value: Any, low: float, high: float) -> float:
    """Convert value to float and clamp it into the inclusive range [low, high].

    Args:
        value: Object to convert and clamp.
        low: Minimum allowed value.
        high: Maximum allowed value.

    Returns:
        A finite float inside [low, high].
    """
    return max(low, min(high, safe_float(value)))


def extract_roads_data(raw_roads_data: Any) -> Optional[Dict[str, Any]]:
    """Extract the actual road dictionary from common BeamNGpy return shapes.

    BeamNGpy examples and versions can expose RoadsSensor data in slightly
    different structures.  Some code sees a direct dictionary:

        {"dist2CL": ..., "roadRadius": ...}

    Other code sees a nested dictionary:

        {0: {"dist2CL": ..., "roadRadius": ...}}

    This helper makes agent.py and rl_agent.py robust to both shapes.

    Args:
        raw_roads_data: Raw object returned by RoadsSensor.poll().

    Returns:
        The dictionary containing road fields, or None if no usable data exists.
    """
    if not isinstance(raw_roads_data, dict) or not raw_roads_data:
        return None

    # Case 1: direct road dictionary.
    if ROAD_KEYS.intersection(raw_roads_data.keys()):
        return raw_roads_data

    # Case 2: dictionary indexed by integer 0.
    if 0 in raw_roads_data and isinstance(raw_roads_data[0], dict):
        nested = raw_roads_data[0]
        if ROAD_KEYS.intersection(nested.keys()):
            return nested

    # Case 3: dictionary indexed by sensor name or another identifier.
    for value in raw_roads_data.values():
        if isinstance(value, dict) and ROAD_KEYS.intersection(value.keys()):
            return value

    return None


def extract_electrics_data(vehicle: Any) -> Dict[str, Any]:
    """Return the Electrics sensor data dictionary from a BeamNG vehicle.

    The project attaches the Electrics sensor under the name "electrics".  In
    BeamNGpy, the readings usually live under vehicle.sensors["electrics"].data.
    This helper returns an empty dictionary instead of crashing when the sensor
    is missing or not ready.

    Args:
        vehicle: BeamNG Vehicle object with attached sensors.

    Returns:
        Electrics data dictionary, or empty dict.
    """
    try:
        electrics = vehicle.sensors["electrics"]
    except Exception:
        return {}

    data = getattr(electrics, "data", None)
    if isinstance(data, dict):
        return data

    # Fallback for mocked tests or alternate object shapes.
    if isinstance(electrics, dict):
        return electrics

    return {}


def roads_and_electrics_to_state(roads_data: Dict[str, Any], electrics_data: Dict[str, Any]) -> List[float]:
    """Convert BeamNG road/electrics readings into the canonical 6 features.

    Args:
        roads_data: Dictionary containing RoadsSensor fields.
        electrics_data: Dictionary containing Electrics fields.

    Returns:
        Model input vector in FEATURE_COLUMNS order:
        [distFromCenter, headingAngle, curvature, roadWidth, drivability, Speed]
    """
    radius = safe_float(roads_data.get("roadRadius", 0.0))
    curvature = 0.0 if radius == 0.0 else 1.0 / radius
    if math.isnan(curvature) or math.isinf(curvature):
        curvature = 0.0

    road_width = 2.0 * safe_float(roads_data.get("halfWidth", 0.0))

    return [
        safe_float(roads_data.get("dist2CL", 0.0)),
        safe_float(roads_data.get("headingAngle", 0.0)),
        curvature,
        road_width,
        safe_float(roads_data.get("drivability", 0.0)),
        safe_float(electrics_data.get("wheelspeed", 0.0)),
    ]


def raw_sensors_to_state(raw_roads_data: Any, vehicle: Any) -> Optional[List[float]]:
    """Convert raw RoadsSensor output plus vehicle electrics into model state.

    Args:
        raw_roads_data: Raw result from RoadsSensor.poll().
        vehicle: BeamNG Vehicle containing the attached Electrics sensor.

    Returns:
        Canonical six-value state list, or None if road data is not usable yet.
    """
    roads_data = extract_roads_data(raw_roads_data)
    if roads_data is None:
        return None

    electrics_data = extract_electrics_data(vehicle)
    return roads_and_electrics_to_state(roads_data, electrics_data)


def row_to_features(row: Any) -> List[float]:
    """Read model features from a pandas row-like object in canonical order.

    Args:
        row: pandas Series or any object supporting row[column].

    Returns:
        List of six floats matching FEATURE_COLUMNS.
    """
    return [safe_float(row[column]) for column in FEATURE_COLUMNS]


def row_to_targets(row: Any) -> List[float]:
    """Read target controls from a pandas row-like object in canonical order.

    Args:
        row: pandas Series or any object supporting row[column].

    Returns:
        List of three floats matching TARGET_COLUMNS.
    """
    return [safe_float(row[column]) for column in TARGET_COLUMNS]


def validate_required_columns(columns: Iterable[str]) -> List[str]:
    """Check whether a CSV/table contains every required training column.

    Args:
        columns: Iterable of column names from a DataFrame or CSV header.

    Returns:
        List of missing required columns.  Empty list means validation passed.
    """
    column_set = set(columns)
    return [column for column in REQUIRED_COLUMNS if column not in column_set]
