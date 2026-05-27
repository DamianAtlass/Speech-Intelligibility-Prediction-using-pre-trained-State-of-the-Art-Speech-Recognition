# Speech-Intelligibility-Prediction-using-pre-trained-State-of-the-Art-Speech-Recognition

## Installation

### ffmpeg
The version needs to be compatible with other libraries, __6.1.1-3ubuntu5__ seems sufficient for this setup. The search for compatible versions on the system seems to happen automatically and doesn't need to be applied manually.
```bash

sudo apt-get install ffmpeg
```
### pip and venv
```bash
python3 -m venv .venv
. .venv/bin/activate 
pip install -r requirements
#optionally for editable, local package:
pip install -e /path/to/sip_whipser/
```

## Setup

### .env

It is required to set up a .env file, the .env_template can be used as a template. Remember to rename the final file.

### pytest

After the venv is set up, it is encouraged to run the __pytest__ command. This will check the functionality
