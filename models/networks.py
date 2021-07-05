import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from models.BANet_model import BANet_model

def get_generator(model_config):
    generator_name = model_config['g_name']
    if generator_name == 'BANet':
        model_g = BANet_model()
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    return nn.DataParallel(model_g)

def get_nets(model_config):
    return get_generator(model_config)
