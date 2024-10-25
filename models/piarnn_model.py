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
    def __init__(self, layers, input_size, hidden_size, output_size, q,dt, rk_weights, eps=1, gamma=1, init_w_std=1, use_gates=False):
        super(AntisymmetricRNN, self).__init__()

        feature_size = 128
        self.layers      = layers

        normal_sample_V  = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([1/feature_size]))
    
        Vh_init_weight   = nn.ParameterList([nn.Parameter(normal_sample_V.sample((hidden_size, input_size))[...,0]) if l < 1 else
                                    nn.Parameter(normal_sample_V.sample((hidden_size, hidden_size))[..., 0])  for l in range(layers)])
        Vh_init_bias     = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for _ in range(layers)])
        self.Vh          = nn.ParameterList([nn.Linear(feature_size, hidden_size) for _ in range(layers)])

        #self.W_out = nn.Parameter(torch.randn(output_size, hidden_size) / np.sqrt(hidden_size))
        self.W_out = nn.ParameterList( [nn.Parameter(torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)) for _ in range(layers)] )
        self.b_out = nn.Parameter(torch.zeros(hidden_size))


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

        # fully connected layer, fc_in1 can be used for feature expansion
        self.fc_in = nn.Linear(input_size, feature_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

        # Initialize loss balancing parameters
        self.log_var_1 = nn.Parameter(torch.zeros(1))
        self.log_var_2 = nn.Parameter(torch.zeros(1))
        self.log_var_3 = nn.Parameter(torch.zeros(1))


    def forward(self, X_):
        X = 2.0 *(X_ - self.lb) / (self.ub - self.lb) - 1.0

        h = torch.zeros(X.shape[0], self.hidden_size)


        T = self.q
        outputs = []
        
        # uniquely for feature expansion, can be removed
        X = self.fc_in(X)

        if not self.use_gates:
            for _ in range(T):
                for layer in range(self.layers):
                    WmWT_h = torch.matmul(h, (self.W[layer] - self.W[layer].transpose(1, 0) - self.gamma_I))
                    Vh_x = self.Vh[layer](X)

                    linear_transform = WmWT_h + Vh_x

                    f = torch.tanh(linear_transform) # torch.tanh
                    h = h + self.eps*f
                outputs.append(h)
         
                
        else:
            for _ in range(T):
                for layer in range(self.layers):
                    WmWT_h    = torch.matmul(h, (self.W[layer] - self.W[layer].transpose(1, 0) - self.gamma_I))
                    Vh_x       = self.Vh[layer](X)
                    
                    # (W - W.T - gammaI)h + Vhx + bh
                    linear_transform_1 = WmWT_h + Vh_x
                    
                    # Vzx + bz
                    Vz_x      = self.Vz[layer](X)
                    linear_transform_2 = WmWT_h + Vz_x 
                    f = torch.tanh(linear_transform_1) * torch.sigmoid(linear_transform_2)
                    h = h + self.eps * f
                outputs.append()
           
        outputs = torch.stack(outputs)
        output = self.fc_out(outputs[-1]) # 0.4078, 0.15

        
        return output

    def predict(self, test_set):
        if type(test_set) != torch.Tensor:
            test_set = torch.from_numpy(test_set).float()
            
        preds = self.forward(test_set)

        return preds.detach().cpu().numpy()
    


class ACPDE(AntisymmetricRNN):
    def __init__(self, layers, input_size, hidden_size, output_size, q, dt, rk_weights, eps=1, gamma=1, init_w_std=1, use_gates=False):
        super(ACPDE, self).__init__(layers, input_size, hidden_size, output_size, q, dt, rk_weights, eps, gamma, init_w_std, use_gates)
        
    

    def fwd_gradients_ac(self, U, X):
        dumx0 = torch.ones([U.shape[0], U.shape[1]], requires_grad=True)
        first_deriv = torch.autograd.grad(U, X,
                                            grad_outputs=dumx0, 
                                            retain_graph=True,
                                            create_graph=True)[0]
        
        sec_deriv_sum = torch.autograd.grad(first_deriv.sum(), 
                                            inputs=dumx0,
                                            create_graph=True)[0]
        return sec_deriv_sum
    
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


    def compute_loss(self, y_train, x_train, bc_=None):
        if bc_ is not None:
            bc = bc_.detach().clone()
            bc.requires_grad = True

        pde_preds, _ = self.ac_eq(x_train)
        bc_preds     = self.forward(bc)
        bc_x         = self.fwd_gradients_ac(bc_preds, bc)

        total_var = torch.exp(self.log_var_1) + torch.exp(self.log_var_2) + torch.exp(self.log_var_3)

        weights1 = 3 * torch.exp(self.log_var_1) / total_var
        weights2 = 2 * torch.exp(self.log_var_2) / total_var
        weights3 = 2 * torch.exp(self.log_var_3) / total_var

        loss1 = torch.sum((y_train - pde_preds)**2)
        loss2 = torch.sum((bc_preds[0, :] - bc_preds[1, :])**2)
        loss3 = torch.sum((bc_x[0,:] - bc_x[1,:])**2)
        loss =  weights1 * loss1 + weights2 + \
                loss2 + weights3 * loss3
        
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param)

        return loss

class BurgersPDE(AntisymmetricRNN):
    def __init__(self, layers, input_size, hidden_size, output_size, q, dt, rk_weights, eps=1, gamma=1, init_w_std=1, use_gates=False):
        super(BurgersPDE, self).__init__(layers, input_size, hidden_size, output_size, q, dt, rk_weights, eps, gamma, init_w_std, use_gates)
        
    
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
    

    def compute_loss(self, y_train, x_train, bc_=None):
        if bc_ is not None:
            bc = bc_.detach().clone()
            bc.requires_grad = True

        pde_preds, _ = self.burger_eq(x_train)
        bc_preds     = self.forward(bc)
        bc_preds_x   = self.fwd_gradients_burger(bc_preds, bc)

        return (0.1 * torch.sum((y_train - pde_preds)**2)) + \
               (0.1 * torch.sum((bc_preds)**2)) #+ (0.5 * torch.sum((bc_preds_x)**2))  
    



def get_PDE(pde_name, *args, **kwargs):
    if pde_name == "ac":
        return ACPDE(*args, **kwargs)
    elif pde_name == "burgers":
        return BurgersPDE(*args, **kwargs)
    else:
        raise ValueError(f"The PDE name: {pde_name} does not exist")
