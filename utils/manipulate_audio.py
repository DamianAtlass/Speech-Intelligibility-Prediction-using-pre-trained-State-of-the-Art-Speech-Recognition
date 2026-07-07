import numpy as np


def get_power(signal):
    return np.mean(signal ** 2)

def calculate_snr(signal1, signal2):
    return 10 * np.log10(get_power(signal1) / get_power(signal2))

def add_gaussion_noise(signal: np.ndarray, target_snr_db: int|float|None) -> np.ndarray:
    """
    Add noise to a signal to resulting in a specific SNR of the returned signal.

    signal, np.ndarray: the original signal
    target_snr_db, float: the SNR which the returned signal should have
    :return:
    """
    if not target_snr_db:
        return signal

    signal_power = get_power(signal)
    noise_power = signal_power / (10 ** (target_snr_db / 10))

    noise = np.random.normal(loc=0, scale=np.sqrt(noise_power), size=len(signal))
    assert np.abs(target_snr_db - calculate_snr(signal, noise)) < 1

    noised_signal = signal + noise

    assert len(signal) == len(noised_signal)

    return noised_signal

def add_noise_transformation(batch: dict):
    batch["audio"] = [
        {"array": add_gaussion_noise(sample["array"], target_snr),
         "sample_rate": sample["sampling_rate"]}
        for sample, target_snr in zip(batch["audio"], batch["snr"])]
    return batch