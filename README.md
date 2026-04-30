# CS450-Self-Driving-Agent

This project collects driving data from BeamNG.tech and trains a simple neural network
to predict steering, throttle, and brake.

----------------

1) REQUIREMENTS

- Python 3.10+ (3.13.7 highly recommended)
- BeamNG.tech, latest version

-----------------------

2) DEPENDENCIES

- PyTorch ("torch" in pip)
- BeamNGpy ("beamngpy" in pip)

No other packages needed.

--------------------------------------

3) USAGE

Start by running the following command:

```
python collect_data.py --beamng-home "(LOCATION OF BEAM.NG)" --frames 200
```

Notes:
- Replace --beamng-home with your actual BeamNG root folder.
- Output is written to raw_data.csv in this repo folder.
- Duration largely depends on computer speed.

After that, normalize the data by running `process_data.py`, no command-line parameters needed.

Then, run `training_loop.ipynb` to train the model.
- Again, duration of training largely depends on your PC's performance.

Finally, run `agent.py` to test the model in-game. Note that this file uses config.json.
Please point the config file to your installation of BeamNG.tech, like before.

----------------------------

4) COMMON ERRORS & SOLUTIONS

No module named beamngpy:
-> python -m pip install beamngpy==1.35

No BeamNG binary found in BeamNG home:
-> Your --beamng-home path is wrong.
   It must point to the BeamNG install root that contains:
   - BeamNG.tech.exe
   OR
   - Bin64/BeamNG.tech.x64.exe

----------------------

5) WHERE DATA IS SAVED

Main collected dataset:
- raw_data.csv
