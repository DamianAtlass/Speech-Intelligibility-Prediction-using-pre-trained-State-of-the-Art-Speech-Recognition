import subprocess as sp
import torch

# mostly from https://stackoverflow.com/questions/67707828/how-to-get-every-seconds-gpu-usage-in-python
def check_gpu_memory_usage(gpu_index_to_check: list = None, threshold: float = 0.1, gpu_mem=None):
    """
    Checks if a GPU's memory usage is above a certain threshold.

    gpu_index_to_check: list, specify what GPUs to check

    threshold: float, specify an upper bound

    gpu_mem: float, specify the GPUs total memory. This should be taken from the nvidia-smi command and needs to be reworked at some point!

    """
    # redo this ugly function

    if gpu_mem is None:
        gpu_mem = [46068, 46068, 46068, 46068]
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    command_executed_successfully = False
    try:
        tmp = sp.check_output(command.split(),stderr=sp.STDOUT)
        memory_use_info = output_to_list(tmp)[1:]
        command_executed_successfully = True
    except FileNotFoundError as e:
        if e.filename == "nvidia-smi":
            print("Nvidia-SMI command not found")
        else:
            raise RuntimeError("Something went wrong.")
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

    if command_executed_successfully:
        memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]

        for i in range(len(memory_use_values)):
            print(f"Memory usage GPU {i}: {memory_use_values[i]}")

        #check usage threshold
        for n in gpu_index_to_check:
            if memory_use_values[n] > gpu_mem[n]*threshold:
                raise RuntimeError(f"GPU {n} busy.")
            else:
                print(f"GPU {n} not busy.")

def manage_device(gpu_index: int = None) -> torch.device:
    if torch.cuda.is_available():
        print("Cuda available")
        if not isinstance(gpu_index, int):
            raise RuntimeError(f"You need to specify a GPU to use. Create an .env and specify 'GPU_DEVICE=_' !")

        check_gpu_memory_usage([gpu_index])
        device = torch.device(f"cuda:{gpu_index}")
    else:
        print("Cuda not available. Use cpu.")
        device = torch.device("cpu")

        return device

    return device

if __name__ == '__main__':
    pass
    #check_gpu_memory_usage([0,1,2,3])