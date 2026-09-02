import os
import torch
import torch.distributed as distr

_BACKEND = "isaac"

_LOCAL_RANK = os.getenv("LOCAL_RANK", "0")
_LOCAL_RANK = int(_LOCAL_RANK)
_RANK = os.getenv("RANK", "0")
_RANK = int(_RANK)
_WORLD_SIZE = os.getenv("WORLD_SIZE", "1")
_WORLD_SIZE = int(_WORLD_SIZE)
_MAIN_PROCESS = _RANK == 0


def is_main_process():
    return _MAIN_PROCESS

def is_distributed():
    return _WORLD_SIZE > 1

def get_local_rank():
    return _LOCAL_RANK

def get_rank():
    return _RANK

def get_world_size():
    return _WORLD_SIZE

def init_distributed():
    if is_distributed():
        torch.cuda.set_device(get_local_rank())
        distr.init_process_group(backend="nccl")

def all_reduce(tensor, op=distr.ReduceOp.SUM):
    if is_distributed():
        distr.all_reduce(tensor, op=op)

def broadcast_object(value):
    if is_distributed():
        values = [value]
        distr.broadcast_object_list(values, src=0)
        return values[0]
    return value

def broadcast_module(module):
    if is_distributed():
        for parameter in module.parameters():
            distr.broadcast(parameter.data, src=0)
        for buffer in module.buffers():
            distr.broadcast(buffer.data, src=0)

def average_gradients(*modules):
    if is_distributed():
        for module in modules:
            for parameter in module.parameters():
                if parameter.grad is not None:
                    distr.all_reduce(parameter.grad)
                    parameter.grad.div_(_WORLD_SIZE)

_print = print
def print(*args, **kwargs):
    _print(f"[RANK {_LOCAL_RANK}/{_WORLD_SIZE}]:", *args, **kwargs)


ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets")

def set_backend(backend: str):
    if not backend in ("isaac", "mujoco"):
        raise ValueError(f"backend must be either 'isaac' or 'mujoco', got {backend}")
    global _BACKEND
    _BACKEND = backend


def get_backend():
    return _BACKEND
