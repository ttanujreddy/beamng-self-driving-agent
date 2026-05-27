================================================================================
  BEAMNG SELF-DRIVING AGENT
  CS450-01 Final Project
  PyTorch Behavior Cloning + Experimental REINFORCE Reinforcement Learning
================================================================================

CONTENTS
================================================================================
  1.  Overview
  2.  Requirements
  3.  Installation
  4.  Pipeline
      4.1  Step 1 - Collect Training Data
      4.2  Step 2 - Process the Data
      4.3  Step 3 - Train the Behavior-Cloning Model
      4.4  Step 4 - Run the Agent in BeamNG
  5.  Experimental Reinforcement Learning Extension
      5.1  Algorithm (REINFORCE)
      5.2  Reward Design
  6.  BeamNG-Free Reward Demo
  7.  File Reference
  8.  Outputs
  9.  Common Errors
  10. Contributions
  11. License & Trademark
================================================================================


================================================================================
  1. OVERVIEW
================================================================================

This project uses BeamNG.tech to build a self-driving car agent.
A small feedforward neural network learns to predict steering, throttle, and
brake from road sensor and electrics data. The main pipeline uses supervised
behavior cloning. An experimental reinforcement-learning extension fine-tunes
the trained model using a REINFORCE-style policy-gradient loop.

  Tech stack:     Python 3.10+, PyTorch, BeamNGpy, Hirochi Raceway (BeamNG.tech)
  State vector:   [distFromCenter, headingAngle, curvature, roadWidth, drivability, Speed]
  Action vector:  [Steering, Throttle, Brake]

[ Back to Contents: see top ]


================================================================================
  2. REQUIREMENTS
================================================================================

  - Python 3.10+ (3.13.7 recommended)
  - BeamNG.tech — available for Windows (experimentally on Linux)
      Download and license: https://beamng.tech
  - Python dependencies listed in `requirements.txt`

[ Back to Contents: see top ]


================================================================================
  3. INSTALLATION
================================================================================

  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt

If BeamNGpy is missing:

  python -m pip install beamngpy==1.35

[ Back to Contents: see top ]


================================================================================
  4. PIPELINE
================================================================================

The pipeline runs in four steps: collect data, process it, train the model,
then run the agent.

--------------------------------------------------------------------------------
  4.1  STEP 1 - COLLECT TRAINING DATA
--------------------------------------------------------------------------------

  python collect_data.py --beamng-home "(LOCATION OF BEAMNG.TECH)" --frames 200

  Example:
  python collect_data.py --beamng-home "C:/BeamNG.tech.v0.38.5.0" --frames 200

  The BeamNG AI drives Hirochi Raceway while road sensor and electrics data
  are logged.

  Arguments:
  ------------------------------------------------------------------------
  | Argument          | Default | Description                            |
  |-------------------|---------|----------------------------------------|
  | --beamng-home     | req.    | Path to BeamNG.tech install folder     |
  | --frames          | 30      | Number of frames to collect            |
  | --with-obstacles  | off     | Add obstacle cars to the scenario      |
  | --startup-delay   | 8.0     | Seconds to wait before first sensor poll|
  | --warmup-polls    | 10      | Warmup polls before the capture loop   |
  | --ai-mode         | span    | BeamNG AI mode: span, traffic, random  |
  | --keep-open       | off     | Wait for Enter before disconnecting    |
  ------------------------------------------------------------------------

  Output:  raw_data.csv

  The collected row format is:

    [distFromCenter, headingAngle, curvature, roadWidth, drivability, Speed,
    Steering, Throttle, Brake]

  The first six values are model inputs. The last three values are control labels.

--------------------------------------------------------------------------------
  4.2  STEP 2 - PROCESS THE DATA
--------------------------------------------------------------------------------

  python process_data.py

  Optional explicit paths:
  python process_data.py --input raw_data.csv --output processed_data.csv

  Validates the CSV schema, drops rows with missing values, clips impossible
  control values.

  Output:  processed_data.csv

--------------------------------------------------------------------------------
  4.3  STEP 3 - TRAIN THE BEHAVIOR-CLONING MODEL
--------------------------------------------------------------------------------

  jupyter notebook training_loop.ipynb

  Trains DrivingModel (6 -> 64 -> 32 -> 3 feedforward network) using MSE loss
  on recorded controls. Uses an 80/10/10 train/val/test split with early
  stopping.

  It learns this mapping:

    state = [distFromCenter, headingAngle, curvature, roadWidth, drivability, Speed]
            ->
    action = [Steering, Throttle, Brake]

  Output:  best_model.pth

--------------------------------------------------------------------------------
  4.4  STEP 4 - RUN THE AGENT IN BEAMNG
--------------------------------------------------------------------------------

  Edit config.json:
  {
    "beamng-path": "(LOCATION OF BEAMNG.TECH)",
    "model-path":  "best_model.pth"
  }

  Then run:
  python agent.py

  The agent loads the trained model, initializes BeamNG, and drives in a
  continuous loop using live sensor data.

[ Back to Contents: see top ]


================================================================================
  5. EXPERIMENTAL REINFORCEMENT LEARNING EXTENSION
================================================================================

After behavior cloning, the RL extension lets the agent improve by acting in
BeamNG and learning from reward feedback — useful for recovering from states
not covered by the training dataset.

  python rl_agent.py

  Optional arguments:
  python rl_agent.py --config config.json
  python rl_agent.py --beamng-home "(LOCATION OF BEAMNG.TECH)"

  To test the RL model, update config.json and run agent.py as normal:
    "model-path": "best_model_rl.pth"

  NOTE:
  REINFORCE is a high-variance Monte Carlo algorithm and training is slow.
  Limited data means the agent can take several minutes of wall time per
  behavior learned (e.g. ~6 minutes to learn not to immediately turn into a
  wall at the start). Improvement is real but incremental: the agent will learn
  to handle the states it encounters most, but will go off-road further down
  the track until given more episodes and data. Treat best_model_rl.pth as a
  work-in-progress candidate rather than a complete replacement for
  best_model.pth.

--------------------------------------------------------------------------------
  5.1  ALGORITHM (REINFORCE)
--------------------------------------------------------------------------------

  1. Load best_model.pth as the starting policy.
  2. Wrap the deterministic model as a stochastic Gaussian policy.
  3. Run BeamNG driving episodes.
  4. Sample actions and record their log probabilities.
  5. Compute rewards using rl_reward.py.
  6. Compute discounted returns.
  7. Update the policy with a REINFORCE-style loss.
  8. Save the best episode model as best_model_rl.pth.

  Output:  best_model_rl.pth

--------------------------------------------------------------------------------
  5.2  REWARD DESIGN
--------------------------------------------------------------------------------

  Rewards:
    Staying near the lane center.
    Alignment with the road heading.
    Maintaining reasonable speed.

  Penalizes:
    Leaving the road or entering non-drivable areas.
    Unstable controls such as unnecessary braking or throttle/brake conflict.

[ Back to Contents: see top ]


================================================================================
  6. BEAMNG-FREE REWARD DEMO
================================================================================

To verify the reward function without BeamNG running:

  python demo_rl_reward.py

This prints reward values for hand-written driving states such as centered, 
near-road-edge, stopped, and off-road cases. This is useful for explaining the 
RL reward design without needing the simulator to run live.

[ Back to Contents: see top ]


================================================================================
  7. FILE REFERENCE
================================================================================

  -------------------------------------------------------------------------
  | File                      | Purpose                                   |
  |---------------------------|-------------------------------------------|
  | collect_data.py           | Runs BeamNG AI, records data to           |
  |                           | raw_data.csv                              |
  | process_data.py           | Cleans and validates raw_data.csv,        |
  |                           | writes processed_data.csv                 |
  | training_loop.ipynb       | Trains the supervised behavior-cloning    |
  |                           | model                                     |
  | model.py                  | Defines DrivingModel, the PyTorch network |
  | utils.py                  | PyTorch Dataset wrapper for driving rows  |
  | state_schema.py           | Shared state/action schema and sensor     |
  |                           | conversion helpers                        |
  | agent.py                  | Runs a trained model live in BeamNG       |
  | rl_reward.py              | Shared RL reward function                 |
  | rl_agent.py               | Experimental REINFORCE fine-tuning loop   |
  | demo_rl_reward.py         | BeamNG-free reward function demonstration |
  | environment_task_setup.py | Scenario layout, spawn positions,         |
  |                           | and obstacle setup                        |
  | dummynet.py               | Dummy network for testing the control     |
  |                           | pipeline without a trained model          |
  | config.json               | BeamNG path, scenario settings,           |
  |                           | and model path                            |
  -------------------------------------------------------------------------

[ Back to Contents: see top ]


================================================================================
  8. OUTPUTS
================================================================================

  -------------------------------------------------------------------------
  | File                | Created by           | Description              |
  |---------------------|----------------------|--------------------------|
  | raw_data.csv        | collect_data.py      | Raw BeamNG driving       |
  |                     |                      | dataset                  |
  | processed_data.csv  | process_data.py      | Cleaned dataset for      |
  |                     |                      | training                 |
  | best_model.pth      | training_loop.ipynb  | Supervised behavior-     |
  |                     |                      | cloning model            |
  | best_model_rl.pth   | rl_agent.py          | Experimental RL          |
  |                     |                      | fine-tuned candidate     |
  -------------------------------------------------------------------------

[ Back to Contents: see top ]


================================================================================
  9. COMMON ERRORS
================================================================================

  ERROR: No module named beamngpy
  FIX:
    python -m pip install beamngpy==1.35

  -------------------------------------------------------------------------

  ERROR: No BeamNG binary found in BeamNG home
  FIX:
    Your --beamng-home path is wrong. It must point to the BeamNG.tech
    install root containing one of:
      BeamNG.tech.exe
      Bin64/BeamNG.tech.x64.exe

  -------------------------------------------------------------------------

  ERROR: raw_data.csv is missing columns
  FIX:
    Collect data using the current collect_data.py. Expected columns:
      distFromCenter, headingAngle, curvature, roadWidth, drivability,
      Speed, Steering, Throttle, Brake

  -------------------------------------------------------------------------

  ERROR: RL training is very slow
  FIX:
    This is expected. Improvement is real but incremental — the agent
    learns to handle states it encounters frequently, but with limited
    training data it can take several minutes per behavior learned.
    Running more episodes and collecting more driving data will extend
    how far down the track the agent can drive reliably. For a quick
    demo, best_model.pth from behavior cloning is the faster path.

[ Back to Contents: see top ]


================================================================================
  10. CONTRIBUTIONS
================================================================================

  -------------------------------------------------------------------------
  | Contributor             | Role / Contributions                        |
  |-------------------------|---------------------------------------------|
  | Matthew Gallenberger    | Data collection pipeline, data processing   |
  |                         | and validation, state schema design         |
  |-------------------------|---------------------------------------------|
  | Brendel Zuniga          | Supervised training loop, RL fine-tuning,   |
  |                         | state schema                                |
  |-------------------------|---------------------------------------------|
  | Gregory Larson          | Live agent inference, dummy network for     |
  |                         | pipeline testing, state schema              |
  |-------------------------|---------------------------------------------|
  | Santiago Ramirez        | Neural network architecture                 |
  |-------------------------|---------------------------------------------|
  | Nathan Fermo            | BeamNG scenario setup, spawn positions,     |
  |                         | and obstacle configuration                  |
  |-------------------------|---------------------------------------------|
  | Tanuj Reddy Thummala    | Training pipeline, RL fine-tuning, reward   |
  |                         | design, reward demo, state schema, and      |
  |                         | documentation                               |
  |-------------------------|---------------------------------------------|
  | Anthony Krauss          | Reward function design, BeamNG-free         |
  |                         | reward demo, and documentation              |
  |-------------------------|---------------------------------------------|
  | Abshir Haybe            | RL agent development, state schema          |
  -------------------------------------------------------------------------

  Additional contributions are noted in the headers of each source file.

[ Back to Contents: see top ]


================================================================================
  11. LICENSE & TRADEMARK
================================================================================

  BeamNG.tech is developed and published by BeamNG GmbH.
  Use of BeamNG.tech requires a valid license: https://beamng.tech

  The BeamNG.tech name and logo are registered trademarks of BeamNG GmbH.
  The logo is reproduced in accordance with BeamNG Trademark Usage Guidelines:
  https://beamng.com/game/support/policies/trademark-guidelines/
  The logo has not been altered in any way other than scaling.

  This project is an independent academic work and is not affiliated with,
  endorsed by, or sponsored by BeamNG GmbH.

[ Back to Contents: see top ]

================================================================================