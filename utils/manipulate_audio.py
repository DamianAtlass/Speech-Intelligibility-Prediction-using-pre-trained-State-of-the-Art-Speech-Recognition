import numpy as np


#def get_power(signal: np.ndarray):
#    return np.mean(signal ** 2)

def get_rms(signal: np.ndarray):
    return np.sqrt(np.mean(signal ** 2))

def calculate_snr(signal: np.ndarray, noise: np.ndarray) ->  np.ndarray:
    return 20 * np.log10(get_rms(signal) / get_rms(noise))

def add_gaussian_noise(signal: np.ndarray, target_snr_db: int | float | None) -> np.ndarray:
    """
    Add noise to a signal to resulting in a specific SNR of the returned signal.

    signal, np.ndarray: the original signal
    target_snr_db, float: the SNR which the returned signal should have
    :return:
    """
    if target_snr_db is None:
        return signal

    signal_rms = get_rms(signal)
    noise_rms = signal_rms / (10 ** (target_snr_db / 20))

    noise = np.random.normal(loc=0, scale=noise_rms, size=len(signal))
    assert np.abs(target_snr_db - calculate_snr(signal, noise)) < 1

    noised_signal = signal + noise

    assert len(signal) == len(noised_signal)

    return noised_signal

def add_noise_transformation(batch: dict):
    batch["audio"] = [
        {"array": add_gaussian_noise(sample["array"], target_snr),
         "sample_rate": sample["sampling_rate"]}
        for sample, target_snr in zip(batch["audio"], batch["snr"])]
    return batch