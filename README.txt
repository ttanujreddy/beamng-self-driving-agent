--------------------------------------------------------------------------------
BEAMNG SELF-DRIVING AGENT
CS450-01 Final Project - PyTorch Behavior Cloning + Experimental REINFORCE RL
--------------------------------------------------------------------------------

LOGO
--------------------------------------------------------------------------------
BeamNG.tech logo used in accordance with BeamNG Trademark Usage Guidelines.
Download the official logo (PNG/SVG) from the BeamNG Media Kit:
https://bng.gg/mediakit (01-Logo folder)
Save as: assets/beamng-tech-logo.png in the repo root.
Trademark guidelines: https://beamng.com/game/support/policies/trademark-guidelines/

OVERVIEW
--------------------------------------------------------------------------------
This project builds a self-driving car agent for the BeamNG.tech simulator.
A neural network predicts steering, throttle, and brake from road sensor data.

Main method:
- Supervised behavior cloning

Extension:
- Experimental reinforcement learning (REINFORCE)

TECH STACK
--------------------------------------------------------------------------------
Python 3.10+ (3.13.7 recommended)
PyTorch
BeamNGpy
BeamNG.tech (Hirochi Raceway)

STATE VECTOR
--------------------------------------------------------------------------------
[distFromCenter | headingAngle | curvature | roadWidth | drivability | Speed]

ACTION VECTOR
--------------------------------------------------------------------------------
[Steering | Throttle | Brake]

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
Python 3.10+ (3.13.7 recommended)
BeamNG.tech (latest version) — available for Windows (experimentally on Linux)
  Download and license: https://beamng.com/tech/

--------------------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------------------
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch torchvision scikit-learn tqdm plotly jupyter

If BeamNGpy missing:
python -m pip install beamngpy==1.35

--------------------------------------------------------------------------------
PIPELINE
--------------------------------------------------------------------------------

STEP 1 - COLLECT DATA
--------------------------------------------------------------------------------
python collect_data.py --beamng-home "C:/BeamNG.tech.v0.38.5.0" --frames 200

DESCRIPTION:
BeamNG AI drives the track while sensors log driving data.

ARGUMENTS
--------------------------------------------------------------------------------
| Argument          | Default | Description                          |
|-------------------|---------|--------------------------------------|
| beamng-home       | req.    | Path to BeamNG install               |
| frames            | 30      | Number of frames to collect          |
| with-obstacles    | off     | Add obstacle cars                    |
| startup-delay     | 8.0     | Wait before polling sensors          |
| warmup-polls      | 10      | Sensor warmup cycles                 |
| ai-mode           | span    | AI mode (span/traffic/random)        |
| keep-open         | off     | Wait before closing                  |

OUTPUT:
raw_data.csv

FORMAT:
distFromCenter | headingAngle | curvature | roadWidth | drivability | Speed |
Steering | Throttle | Brake

--------------------------------------------------------------------------------
STEP 2 - PROCESS DATA
--------------------------------------------------------------------------------
python process_data.py

OPTIONAL:
python process_data.py --input raw_data.csv --output processed_data.csv

DESCRIPTION:
- Validates schema
- Drops missing rows
- Clips invalid control values

OUTPUT:
processed_data.csv

--------------------------------------------------------------------------------
STEP 3 - TRAIN MODEL
--------------------------------------------------------------------------------
jupyter notebook training_loop.ipynb

DESCRIPTION:
- Feedforward network (6 -> 64 -> 32 -> 3)
- MSE loss
- 80/10/10 split
- Early stopping

OUTPUT:
best_model.pth

--------------------------------------------------------------------------------
STEP 4 - RUN AGENT
--------------------------------------------------------------------------------
Edit config.json:

{
  "beamng-path": "C:/BeamNG.tech.v0.38.5.0",
  "model-path": "best_model.pth"
}

Run:
python agent.py

DESCRIPTION:
Loads model and drives in BeamNG using live sensor data.

--------------------------------------------------------------------------------
REINFORCEMENT LEARNING (EXPERIMENTAL)
--------------------------------------------------------------------------------

RUN RL:
--------------------------------------------------------------------------------
python rl_agent.py

OPTIONAL:
python rl_agent.py --config config.json
python rl_agent.py --beamng-home "C:/BeamNG.tech.v0.38.5.0"

--------------------------------------------------------------------------------
ALGORITHM (REINFORCE)
--------------------------------------------------------------------------------
1. Load best_model.pth
2. Convert to stochastic Gaussian policy
3. Run BeamNG episode
4. Sample actions
5. Compute reward
6. Compute discounted return (gamma = 0.99)
7. Update weights: loss = -log_prob * return
8. Save model

OUTPUT:
best_model_rl.pth

--------------------------------------------------------------------------------
REWARD DESIGN
--------------------------------------------------------------------------------
| Signal                    | Effect                               |
|---------------------------|--------------------------------------|
| speed * cos(headingAngle) | Reward forward motion                |
| distance from center      | Penalize lane deviation              |
| curvature * speed         | Penalize fast turns                  |
| throttle + brake          | Penalize conflict                    |
| steering magnitude        | Penalize sharp turns                 |
| off-road                  | Large penalty + terminate            |
| stopped                   | Small penalty                        |

NOTE:
REINFORCE is high variance and slow.
Model improvement is incremental.
best_model_rl.pth is NOT guaranteed better than best_model.pth.

--------------------------------------------------------------------------------
REWARD DEMO (NO BEAMNG)
--------------------------------------------------------------------------------
python demo_rl_reward.py

DESCRIPTION:
Shows reward outputs for test scenarios.

--------------------------------------------------------------------------------
FILES
--------------------------------------------------------------------------------
| File                      | Purpose                              |
|---------------------------|--------------------------------------|
| collect_data.py           | Collect driving data                 |
| process_data.py           | Clean dataset                        |
| training_loop.ipynb       | Train model                          |
| model.py                  | Neural network definition            |
| utils.py                  | Dataset wrapper                      |
| state_schema.py           | Feature/target schema                |
| agent.py                  | Run trained model                    |
| rl_reward.py              | Reward function                      |
| rl_agent.py               | RL training loop                     |
| demo_rl_reward.py         | Reward demo                          |
| environment_task_setup.py | Scenario setup                       |
| reward_logger.py          | Log rewards                          |
| config.json               | Configuration                        |

--------------------------------------------------------------------------------
OUTPUT FILES
--------------------------------------------------------------------------------
| File                 | Created by             | Description                |
|----------------------|------------------------|----------------------------|
| raw_data.csv         | collect_data.py        | Raw data                   |
| processed_data.csv   | process_data.py        | Clean data                 |
| best_model.pth       | training               | Supervised model           |
| best_model_rl.pth    | rl_agent.py            | RL model (experimental)    |

--------------------------------------------------------------------------------
COMMON ERRORS
--------------------------------------------------------------------------------

ERROR: No module named beamngpy
FIX:
python -m pip install beamngpy==1.35

ERROR: No BeamNG binary found
FIX:
Check beamng-home path:
BeamNG.tech.exe
or
Bin64/BeamNG.tech.x64.exe

ERROR: Missing CSV columns
FIX:
Re-run collect_data.py

EXPECTED COLUMNS:
distFromCenter | headingAngle | curvature | roadWidth | drivability | Speed |
Steering | Throttle | Brake

ERROR: RL training slow
FIX:
Normal behavior. RL takes time.

--------------------------------------------------------------------------------
CONTRIBUTIONS
--------------------------------------------------------------------------------
| Name                       | Contribution                         |
|----------------------------|--------------------------------------|
| Matthew Gallenberger       | Data collection + processing         |
| Brendel Zuniga             | Training + RL                        |
| Gregory Larson             | Agent                                |
| Santiago Ramirez           | Model                                |
| Nathan Fermo               | Environment setup                    |
| Tanuj Reddy Thummala       | RL + reward system                   |
--------------------------------------------------------------------------------

Additional contributions are noted in the headers of each source file.

--------------------------------------------------------------------------------
LICENSE & TRADEMARK
--------------------------------------------------------------------------------
BeamNG.tech is developed and published by BeamNG GmbH.
Use of BeamNG.tech requires a valid license: https://beamng.com/tech/

The BeamNG.tech name and logo are registered trademarks of BeamNG GmbH.
The logo is reproduced in accordance with BeamNG Trademark Usage Guidelines:
https://beamng.com/game/support/policies/trademark-guidelines/
The logo has not been altered in any way other than scaling.

This project is an independent academic work and is not affiliated with,
endorsed by, or sponsored by BeamNG GmbH.
--------------------------------------------------------------------------------