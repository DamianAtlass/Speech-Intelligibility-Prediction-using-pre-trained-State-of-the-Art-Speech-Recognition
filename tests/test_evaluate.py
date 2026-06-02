from evaluate_run import get_data
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
import pytest

def test_get_data():
    pass