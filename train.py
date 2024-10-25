import os
import sys 
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
plt.rcParams['text.usetex'] = False
import torch
import torch.nn as nn
import time 
import scipy.io 
from models.piarnn_model import AntisymmetricRNN, ACPDE, BurgersPDE, get_PDE
from utilities.utils import load_data, apply_lower_triangular_mask, apply_upper_triangular_mask

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False






@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    try:
        set_seed(43)

        # load the RK weights
        rk_weights = cfg.weight_path

        x_train, y_train, x_test, y_test, bc, dt = load_data(data_path=cfg.data_path, pde_name=cfg.pde.name)

        # 
        train_loss       = []
        train_loss_lbfgs = []

        # params
        layers      = cfg.params.layers[1]
        adam_lr     = cfg.adam_params.learning_rate[2]
        adam_epochs = cfg.adam_params.epochs
        lbfgs_epochs= cfg.lbfgs_params.epochs
        lbfg_lr     = cfg.lbfgs_params.learning_rate
        max_it      = cfg.lbfgs_params.max_iter
        max_eval    = cfg.lbfgs_params.max_eval


        input_size  = cfg.net.input_size[0]
        hidden_size = cfg.net.hidden_size[1]
        output_size = cfg.net.output_size + 1
        q           = cfg.rk_weights.q_value

        model = get_PDE(cfg.pde.name, layers, input_size, hidden_size, output_size, q,dt, rk_weights=rk_weights, use_gates=False)
        # print(f"model: {model}")
        # sys.exit()
        optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
        lbfgs_optimizer = torch.optim.LBFGS(
                  model.parameters(),
                  lr=lbfg_lr,
                  max_iter=max_it,
                  max_eval=max_eval,
                  tolerance_grad=1e-7,
                  tolerance_change= 1.0 * np.finfo(float).eps, # 1e-9
                  history_size = 50,
                  line_search_fn="strong_wolfe") 
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.1)


        start_time = time.time()
        for epoch in tqdm(range(adam_epochs)): # num_epochs
            optimizer.zero_grad()
            loss = model.compute_loss(y_train, x_train, bc)
            
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            apply_upper_triangular_mask(model.W)
            #apply_lower_triangular_mask(model.W)
            
            optimizer.step()
            # scheduler.step()

            train_loss.append(loss.item())

            current_lr = optimizer.param_groups[0]['lr']

            
            if epoch%100==0:
                elapsed = time.time() - start_time
                print(f"Epoch :{epoch}, loss: {loss.item():.3e}, elapsed_time: {elapsed}, current lr:{current_lr}")

        # Include a second optimizer
        print("Switched to the second optimizer: LBFG-S")

        # for epoch in range(lbfgs_epochs):

        #     def closure(x_train=x_train, bc=bc):
        #         lbfgs_optimizer.zero_grad()
        #         if cfg.pde.name == "ac":
        #             pde_preds, h_t = model.ac_eq(x_train)
        #             bc_preds, bc_x  = model.bc(bc)
        #             loss  = torch.sum((y_train - pde_preds)**2) + torch.sum((bc_preds[0, :] - bc_preds[1, :])**2) +\
        #             torch.sum((bc_x[0,:] - bc_x[1,:])**2)
                
        #         elif cfg.pde.name == "burgers":
        #                 pde_preds, h_t = model.burger_eq(x_train)
        #                 bc_preds, _  = model.bc(bc)
        #                 loss  = torch.sum((y_train - pde_preds)**2) + torch.sum((bc_preds)**2)
                
        #         else:
        #             raise ValueError(f"This PDE name: {cfg.pde.name} does not exist")

        #         loss.backward()
        #         #nn.utils.clip_grad_norm_(model.parameters(), 1.5)

        #         print(f"The loss: {loss.item()}")
            
        #         return loss
            
        #     lbfgs_optimizer.step(closure)

        #     train_loss_lbfgs.append(loss.item())

        end_time = time.time() - start_time
        print(f"End Time {end_time / 60}")


        # Test the model
        preds = model.predict(x_test)
        error = np.linalg.norm(preds[:, -1] - y_test, 2)/np.linalg.norm(y_test, 2)
        print(f"Error on the test sample: {error}")

        # for i in range(preds.shape[1]):
        #     error = np.linalg.norm(preds[:, -i] - y_test, 2)/np.linalg.norm(y_test, 2)
        #     print(f"Error on the test sample: {error}")

    except KeyError as e:
        raise ValueError(f"Incorrect key error: {e}. Check the YAML configuration file.")
    

if __name__ == "__main__":
    main()