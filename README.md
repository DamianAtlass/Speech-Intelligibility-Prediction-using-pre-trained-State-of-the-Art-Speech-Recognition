# Speech-Intelligibility-Prediction-using-pre-trained-State-of-the-Art-Speech-Recognition

## Installation

### Needed packages

#### ffmpeg
The version needs to be compatible with other libraries, __6.1.1-3ubuntu5__ seems sufficient for this setup. The search for compatible versions on the system seems to happen automatically and doesn't need to be applied manually.
```bash

#### pip

#### nvidia-smi
On Cerberus:
NVIDIA-SMI version  : 580.159.03
NVML version        : 580.159
DRIVER version      : 580.159.03
CUDA Version        : 13.0

On Sontra:
NVIDIA-SMI version  : 595.71.05
NVML version        : 595.71
DRIVER version      : 595.71.05
CUDA Version        : 13.2

## nvidia-smi
On Cerberus:
version 3.0.2


On Sontra:
?

sudo apt-get install ffmpeg
```
### pip and venv

Self explanitory.


```bash
python3 -m venv .venv
. .venv/bin/activate 
pip install -r requirements.txt
#optionally for editable, local package:
pip install -e /path/to/sip_whipser/
```

## Setup

### .env

It is required to set up a .env file, the .env_template can be used as a template. Remember to rename the final file.

### pytest

To check the functionality, first the file __my_pytest__ needs to be made executable:
```bash
chmod +x my_pytest
```
The file exists, because pytest that run cuda won't run in a chain and will get stuck.

Run the tests with:
```bash
bash my_pytest
```
Some tests may get stuck if the machine it runs on is connected the internet via a VPN.
