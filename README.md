# opencv_autonomous_vehicles

This repository presents some classic Computer Vision techniques and applies them to autonomous vehicle perception problem.
* Camera calibration
* Ego lane detection

### Dependencies
* [OpenCV](https://opencv.org)
* Python 3.9
 
### Python
Create virtual environment with Python 3.9.9, activate it, and install libraries.
```
python3 -m venv .venv  
source .venv/bin/activate
pip install -r requirements.txt
pip install ipykernel
python3 -m ipykernel install --user --name=.venv
```
<!-- 
Make sure you have opencv installed. This can be done on macOS with Homebrew:
```
brew install opencv
``` -->