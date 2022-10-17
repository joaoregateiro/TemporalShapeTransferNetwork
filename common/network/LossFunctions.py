''' Util class for deep learning common functions
Author: Joao Regateiro
'''

# System imports
import numpy as np

# Torch imports
import torch
from torch.nn import functional as F



def bce_loss_function(recon_x, x):
    loss = F.binary_cross_entropy(recon_x, x)
    return loss

def convert(M):
    """
    input: M is Scipy sparse matrix
    output: pytorch sparse tensor in GPU
    """
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long().cuda()
    values = torch.from_numpy(M.data).cuda()
    shape = torch.Size(M.shape)
    Ms = torch.sparse_coo_tensor(indices, values, shape, device=torch.device('cuda'))
    return Ms

