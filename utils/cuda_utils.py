import subprocess as sp
import torch
import logging
import os
logger = logging.getLogger(__name__)

# mostly from https://stackoverflow.com/questions/67707828/how-to-get-every-seconds-gpu-usage-in-python
def check_gpu_memory_usage(gpu_index: int, threshold: float = 0.05, gpu_mem=None):
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
            logger.info("Nvidia-SMI command not found")
        else:
            raise RuntimeError("Something went wrong.")
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

    if command_executed_successfully:
        memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]

        for i in range(len(memory_use_values)):
            logger.info(f"Memory usage GPU {i}: {memory_use_values[i]} MiB")

        #check usage threshold

        if memory_use_values[gpu_index] > gpu_mem[gpu_index]*threshold:
            raise RuntimeError(f"GPU {gpu_index} busy.")
        else:
            logger.info(f"Usage of GPU {gpu_index} is under {threshold}%.")
#os.environ["DEVICE_HAS_USABLE_GPU"]
def get_gpu_index() -> int:
    if (gpu_index:=os.getenv("CUDA_VISIBLE_DEVICES", None)) is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not set! Create a .env with 'CUDA_VISIBLE_DEVICES=[GPU_ID_HERE]'or pass it via thecommand line.")
    gpu_index = int(gpu_index)

    return gpu_index

def select_device() -> torch.device:

    gpu_index = get_gpu_index()

    if torch.cuda.is_available():
        logger.info("Cuda available")

        check_gpu_memory_usage(gpu_index)
        device = torch.device("cuda")
        # cuda will automatically use index CUDA_VISIBLE_DEVICES from now
        # this is needed for functions, which use cuda automatically and won't allow specific device asignment
        logger.info(f"Use GPU with index {gpu_index}", )

    else:
        logger.info("Cuda not available. Use cpu.")
        device = torch.device("cpu")

    return device

if __name__ == '__main__':
    pass
    #check_gpu_memory_usage([0,1,2,3])