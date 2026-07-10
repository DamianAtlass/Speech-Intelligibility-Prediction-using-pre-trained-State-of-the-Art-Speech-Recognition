from torch._dynamo import exc

from utils.manipulate_audio import calculate_snr, add_gaussian_noise
import numpy as np
import pytest

@pytest.mark.parametrize(("power_diff", "expected_snr"),[
    (1, 0),
    (0.5, 6),
    (0.1, 20),
])
def test_calculate_snr(power_diff: int, expected_snr: int):
    s = np.random.normal(loc=0, scale=1, size=5000)
    n = np.random.normal(loc=0, scale=power_diff, size=5000)

    snr = calculate_snr(s, n)
    assert expected_snr == np.round(snr, decimals=0)

def test_add_gaussian_noise():
    s = np.random.normal(0, 0.5, 5000)
    r = add_gaussian_noise(s, 7)

    noise = r - s
    snr = calculate_snr(s, noise)
    print(snr)