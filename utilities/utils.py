import os 
import sys 
from typing import Tuple, List, Sequence, Literal, Optional
import scipy.io 
import scipy 
import numpy as np
import torch 


def load_data(data_path, pde_name, N:int = 200) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    data = scipy.io.loadmat(data_path)

    # Boundary conditions
    lb   = torch.tensor([-1.0])
    ub   = torch.tensor([1.0])

    if pde_name == "ac":
        idx_t0 = 20
        idx_t1 = 180

        t      = data["tt"].flatten()[:, None] # Tx 1, (201, 1)
        x      = np.real(data["x"].flatten()[:, None]) # N x 1, (512, 1)
        Exact  = np.real(data["uu"]).T # T x N, (201, 512)

    elif pde_name == "burgers":
        idx_t0 = 10
        idx_t1 = 90

        t      = data["t"].flatten()[:, None]
        x      = np.real(data["x"].flatten()[:, None])
        Exact  = np.real(data["usol"]).T

    else:
        raise ValueError(f"Unkown PDE name: {pde_name}")

    dt = np.array(t[idx_t1] - t[idx_t0])
    x_star = x
    

    noise_u0 = 0.0 
    idx_x = np.random.choice(Exact.shape[1], N, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact[idx_t0 : idx_t0 + 1, idx_x].T
    u0 = u0 + noise_u0 * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])

    x_train = torch.from_numpy(x0).float()
    y_train = torch.from_numpy(u0).float()
    
    x_test  = torch.from_numpy(x_star).float()
    y_test  = np.array(Exact[idx_t1, :])
    
    dt      = torch.from_numpy(dt).float()
    bc = torch.from_numpy(np.vstack((lb,ub))).float()

    #print(f"Y test shape: {y_test.shape}, x_test shape: {x_test.shape}")
    #sys.exit()

    return x_train, y_train, x_test, y_test, bc, dt




# Apply a mask to keep the upper triangular property
def apply_upper_triangular_mask(params):
    for p in params:
        if p.requires_grad:
            mask = torch.triu(torch.ones(p.size(), dtype=torch.bool))
            p.grad.data *= mask

def apply_lower_triangular_mask(params):
    for p in params:
        if p.requires_grad:
            mask = torch.tril(torch.ones(p.size(), dtype=torch.bool))
            p.grad.data *= mask