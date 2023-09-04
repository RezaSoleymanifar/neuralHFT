import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import gelu
import math






class SA_config:
    def __init__(
        self,
        vocab_size = 9000,
        embedding = 1200, 
        n_head = 25,
        n_layers = 13,
        ACT = gelu,
        bias = True
        ):
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.n_head = n_head
        self.mlp_intermediate = 4 * embedding
        self.n_layers = n_layers
        self.ACT = ACT
        self.bias = bias
