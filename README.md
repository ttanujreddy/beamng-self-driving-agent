# CS450-Self-Driving-Agent

This project collects driving data from BeamNG.tech and trains a simple neural network
to predict steering, throttle, and brake.

1) REQUIREMENTS
----------------
- Python 3.10+
- BeamNG.tech

2) INSTALL DEPENDENCIES
-----------------------
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch torchvision scikit-learn tqdm plotly jupyter

3) COLLECT TRAINING DATA
--------------------------------------
python collect_data.py --beamng-home "(LOCATION OF BEAM.NG)" --frames 200

Notes:
- Replace --beamng-home with your actual BeamNG root folder.
- Output is written to raw_data.csv in this repo folder.
- Duration largely depends on computer speed.

4) OPTIONAL TRAINING NOTEBOOK
-----------------------------
jupyter notebook training_loop.ipynb

5) COMMON ERRORS & SOLUTIONS
----------------------------
No module named beamngpy:
-> python -m pip install beamngpy==1.35

No BeamNG binary found in BeamNG home:
-> Your --beamng-home path is wrong.
   It must point to the BeamNG install root that contains:
   - BeamNG.tech.exe
   OR
   - Bin64/BeamNG.tech.x64.exe

6) WHERE DATA IS SAVED
----------------------
Main collected dataset:
- raw_data.csv
