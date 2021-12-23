r"""
A set of modules to plugin into Megatron-LM with MoE
"""
from .utils import add_moe_args

from .checkpoint import save_checkpoint
from .checkpoint import load_checkpoint
