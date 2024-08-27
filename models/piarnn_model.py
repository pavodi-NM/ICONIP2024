import sys 
import abc
import torch 
import torch.nn as nn 
import numpy as np 
from typing import List, Callable, Optional


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(43)

# activations
silu = nn.SiLU()


class AntisymmetricRNN(nn.Module):
    def __init__(self, layers, input_size, hidden_size, output_size, q,dt, rk_weights, eps=0.01, gamma=0.01, init_w_std=1, use_gates=False):
        super(AntisymmetricRNN, self).__init__()

        self.layers      = layers

        normal_sample_V  = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([1/input_size]))
    
        Vh_init_weight   = nn.ParameterList([nn.Parameter(normal_sample_V.sample((hidden_size, input_size))[...,0]) if l < 1 else
                                    nn.Parameter(normal_sample_V.sample((hidden_size, hidden_size))[..., 0])  for l in range(layers)])
        Vh_init_bias     = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for _ in range(layers)])
        self.Vh          = nn.ParameterList([nn.Linear(input_size, hidden_size) for _ in range(layers)])


        if use_gates:
            self.Vz = nn.ParameterList([nn.Linear(input_size, hidden_size) for _ in range(layers)]) 

        # Initialize w
        normal_sample_w      = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([init_w_std/hidden_size]))
        sampling             = normal_sample_w.sample((hidden_size, hidden_size))[...,0]
        sampling_upper_tri   = torch.triu(sampling, diagonal=1)
        self.W               = nn.ParameterList([nn.Parameter(sampling_upper_tri) for _ in range(layers)])

        # init diffusion
        self.gamma_I     = torch.eye(hidden_size, hidden_size) * gamma
        self.eps         = eps
        self.use_gates   = use_gates
        self.hidden_size = hidden_size
        

        # lower bound and uppper bound
        self.lb  = torch.tensor([-1.0])
        self.ub  = torch.tensor([1.0])


        self.q   = q
        self.dt  = dt

        # Runge-Kutta Weights       
        tmp = np.float32(np.loadtxt(rk_weights, ndmin = 2))
        self.IRK_weights =  torch.tensor(np.reshape(tmp[0:q**2+q], (q+1,q)), dtype=torch.float32)    
        self.IRK_times = tmp[q**2+q:]

        # fully connected layer
        # self.fc1 = nn.Linear(hidden_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)


    def forward(self, X_):
        X = 2.0 *(X_ - self.lb) / (self.ub - self.lb) - 1.0

        h = torch.zeros(X.shape[0], self.hidden_size)

        T = self.q

        if not self.use_gates:
            for _ in range(T):
                for layer in range(self.layers):
                    WmWT_h = torch.matmul(h, (self.W[layer] - self.W[layer].transpose(1, 0) - self.gamma_I))
                    # Vhx + bh
                    Vh_x = self.Vh[layer](X)

                    linear_transform = WmWT_h + Vh_x
                    f = torch.tanh(linear_transform) # torch.tanh
                    h = h + self.eps*f
                
        else:
            for _ in range(T):
                for layer in range(self.layers):
                    WmWT_h    = torch.matmul(h, (self.W[layer] - self.W[layer].transpose(1, 0) - self.gamma_I))
                    Vh_x       = self.Vh[layer](X)
                    
                    # (W - W.T - gammaI)h + Vhx + bh
                    linear_transform_1 = WmWT_h + Vh_x
                    
                    # Vzx + bz
                    Vz_x      = self.Vz[layer](X)
                    
                    # (W - W.T - gammaI)h + Vz_x 
                    linear_transform_2 = WmWT_h + Vz_x 
                    
                    # Tanh
                    f = torch.tanh(linear_transform_1) * torch.sigmoid(linear_transform_2)
                    
                    # output 
                    h = h + self.eps * f
           

        #print(f"H size {h.shape}")
        #sys.exit()
        #output = torch.tanh(self.fc1(h))
        #output = torch.tanh(self.fc2(output))

        #print(f"Output size: {output.shape}")
        #sys.exit()
        output = self.fc3(h)
        
        return output

    def fwd_gradients_ac(self, U, X):
        dumx0 = torch.ones([U.shape[0], U.shape[1]], requires_grad=True)
        first_deriv = torch.autograd.grad(U, X,
                                          grad_outputs=dumx0, 
                                          retain_graph=True,
                                          create_graph=True)[0]
        
        first_deriv_sum = torch.autograd.grad(first_deriv.sum(), 
                                            inputs=dumx0,
                                           create_graph=True)[0]
        return first_deriv_sum


    def fwd_gradients_1(self, U, X):
        dumx1 = torch.ones([U.shape[0], U.shape[1]], requires_grad=True)
        first_deriv    = torch.autograd.grad(U, X,
                                          grad_outputs = dumx1,
                                          retain_graph=True,
                                          create_graph=True,
                                         )[0]
        first_deriv_sum = torch.autograd.grad(outputs=first_deriv.sum(), 
                                              inputs=dumx1,
                                              create_graph=True)[0]

        return first_deriv_sum
    
    def ac_eq(self, X_):

        X     = X_.clone().requires_grad_(True)
        h_t1   = self.forward(X)
        h_t   = h_t1[:, :-1]

        h_tx  = self.fwd_gradients_ac(h_t, X)  # First derivative 
        h_txx = self.fwd_gradients_ac(h_tx, X) # second derivative

        pde   = 5.0*h_t-5.0*h_t**3+0.0001*h_txx
        
        # the weighted sum of the RK
        output= h_t1 - self.dt *(torch.matmul(pde, self.IRK_weights.T))

        return output, h_t1
    

    def fwd_gradients_burger(self, U, x):
        dummy_var   = torch.ones([U.shape[0], U.shape[1]], requires_grad=True)
        first_deriv = torch.autograd.grad(U, x,
                                            grad_outputs=dummy_var, 
                                            retain_graph=True,
                                            create_graph=True)[0]
        second_deriv = torch.autograd.grad(first_deriv.sum(), 
                                            dummy_var,
                                            create_graph=True)[0]
        return second_deriv

    
    def burger_eq(self, X_):
        nu    = 0.01/np.pi 
        X     = X_.clone().requires_grad_(True)
        h_t1   = self.forward(X)
        h_t   = h_t1[:, :-1]

        h_tx  = self.fwd_gradients_burger(h_t, X)  # First derivative 
        h_txx = self.fwd_gradients_burger(h_tx, X) # second derivative

        pde   = -h_t*h_tx + nu * h_txx
        output= h_t1 - self.dt *(torch.matmul(pde, self.IRK_weights.T))

        return output, h_t1

    def bc(self, X_):
        X = X_.detach().clone()
        X.requires_grad = True
        h_t = self.forward(X)

        h_tx  = self.fwd_gradients_1(h_t, X)
        
        return h_t, h_tx
    
    def residual_loss(self):
        pass

    def predict(self, test_set):
        if type(test_set) != torch.Tensor:
            test_set = torch.from_numpy(test_set).float()

        data = test_set.clone().requires_grad_(True)
            
        preds = self.forward(data)

        return preds.detach().cpu().numpy()
    


class ACStrategy(AntisymmetricRNN):
    def forward(self, model, x_train, bc):
        pde_preds, _ = model.ac_eq(x_train)
        bc_preds, bc_x = model.bc(bc)
        return pde_preds, bc_preds, bc_x

    def compute_loss(self, y_train, pde_preds, bc_preds, bc_x):
        return torch.sum((y_train - pde_preds)**2) + \
               torch.sum((bc_preds[0, :] - bc_preds[1, :])**2) + \
               torch.sum((bc_x[0,:] - bc_x[1,:])**2)

class BurgersStrategy(AntisymmetricRNN):
    def forward(self, model, x_train, bc):
        pde_preds, _ = model.burger_eq(x_train)
        bc_preds, _ = model.bc(bc)
        return pde_preds, bc_preds, None

    def compute_loss(self, y_train, pde_preds, bc_preds, bc_x=None):
        return torch.sum((y_train - pde_preds)**2) + \
               torch.sum((bc_preds)**2) / torch.norm(y_train, p=2)
    


