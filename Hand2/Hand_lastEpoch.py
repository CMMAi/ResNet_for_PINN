# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:12:49 2023

@author: MI
"""

import matplotlib.ticker as ticker
# import RHS
import torch
import torch.nn as nn                     # neural networks
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

import numpy as np
import time
import scipy.io
import warnings
warnings.filterwarnings("ignore")

#Set default dtype to float32
torch.set_default_dtype(torch.float)

#PyTorch random number generator
torch.manual_seed(12)#12,12

# Random number generators in other libraries
np.random.seed(12)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)   

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 

## Adam Optimizer
steps=0 #0,0
lr=1e-3

## LBFGS Optimizer
steps2=50000 #50000,50000
lr2=1e-1

#ns: Number of training points 
ns = 3000 #3000,3000
ni = 500 #700,500

my_list = [3] + [30] * 5 + [1] #30*5,30*5
layers = np.array(my_list)

data = scipy.io.loadmat(r"G:\My Drive\ColabNotebooks\Juan\PINN-Interpolation\simple NN\skipp connection\Adam\3D\Hand2\hand2.mat") 
data_in = scipy.io.loadmat(r"G:\My Drive\ColabNotebooks\Juan\PINN-Interpolation\simple NN\skipp connection\Adam\3D\Hand2\hand2_int.mat") 


## all points
dsites0 = data['hand2'] # array
intnode0 = data_in['hand2_int']

dsites = torch.tensor(dsites0) #tensor
intnode = torch.tensor(intnode0)




##############################################################
rhs_sur =  torch.zeros(dsites.shape[0], 1)
rhs_in = torch.ones(intnode.shape[0],1)#x0**2+ y0**2 + z0**2 #
# rhs_in = intnode[:,0]**2+ intnode[:,1]**2 + intnode[:,2]**2 
rhs_in = torch.tensor(rhs_in)
rhs_in = rhs_in.view(-1,1)
rhs_in = rhs_in.float()


############################################################
############################################################
### select randomly
#surface
idx0=np.random.choice(dsites.shape[0],ns,replace=False)
dsites_ns =dsites[idx0,:]
rhs_sur_ns = rhs_sur[idx0,:]
#interior
idx1=np.random.choice(intnode.shape[0],ni,replace=False)
intnode_ni =intnode[idx1,:]
rhs_in_ni =  rhs_in[idx1,:]
# surf+int
xy_train_Nu = torch.cat((intnode_ni, dsites_ns), dim=0)
u_train_Nu = torch.cat((rhs_in_ni, rhs_sur_ns), dim=0)

###########################################################
############################################################

grid_data = scipy.io.loadmat(r"G:\My Drive\ColabNotebooks\Juan\PINN-Interpolation\simple NN\skipp connection\Adam\3D\Hand2\griddata.mat") 

X_test0 = grid_data['epoints']     
X_test = torch.tensor(X_test0)                              # 256 points between -1 and 1 [256x1]


############################################################
############################################################
iter = 0
lossLBFGS = []
errorLBFGS = []

class FCN(nn.Module):
  #https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks/tree/main/PyTorch/Burgers'%20Equation
    ##Neural Network
    def __init__(self,layers, skip_connection=True):
        super().__init__() #call __init__ from parent class 
        'activation function'
        self.activation0 = nn.Tanh() #nn.ReLU() #
        self.activation1 = nn.ReLU() #nn.Tanh() #
        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')#L1Loss() #
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]) 
        self.iter = 0 #For the Optimizer
        self.skip_connection = skip_connection

        'Xavier Normal Initialization'
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)   
    'foward pass'
    def forward(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        a = x.float()
        for i in range(len(layers)-2):  
            z = self.linears[i](a)   
            a = self.activation0(z)   
            # a = (z) +  a
            if (i) % 2 == 0:  # Skip connection every 2 layers
                a = (z)**2 +  a #(z + a)  # Add skip connectionz +    
        a = self.linears[-1](a)
        return a
    'Loss Functions'
    #Loss BC
    def lossBC(self,x_BC,y_BC):
      loss_BC=self.loss_function(self.forward(x_BC),y_BC)
      return loss_BC

    def loss(self,x_BC,y_BC):
      loss_bc=self.lossBC(x_BC,y_BC)
      return loss_bc
    def closure(self):
        optimizer.zero_grad()
        loss = self.loss(xy_train_Nu, u_train_Nu)
        loss.backward()
        global iter  # Reference the global variable
        # err = PINN.test()
        lossLBFGS.append(loss.detach().cpu().numpy())
        # errorLBFGS.append(err.detach().cpu().numpy())
        iter += 1
        if iter % 500 == 0:
            # error_vec = PINN.test()
            print(iter,loss.detach().cpu().numpy())
        return loss        
    'test neural network'
    def test(self):
        u_pred = self.forward(X_test)
        error_vec = torch.tensor([1.0])# torch.linalg.norm((U_test-u_pred),2)/torch.linalg.norm(U_test,2)#  torch.linalg.norm((U_test-u_pred),2)/Nu  #    # Relative L2 Norm of the error (Vector)
        u_pred = u_pred.cpu().detach().numpy()
        # u_pred = np.reshape(u_pred,(n0,n0),order='F')
        return error_vec



################################################

print("Total test  points:",X_test.shape)


#Create Model
PINN = FCN(layers, skip_connection=True)
PINN.to(device)
# print(PINN)

#################################
##Adam
params = list(PINN.parameters())

start_time_A = time.time()
optimizer = torch.optim.Adam(PINN.parameters(),lr=lr,amsgrad=False)

lossADAM = np.zeros(steps)
errorADAM = np.zeros(steps)
for i in range(steps):
    if i==0:
      print("Training Loss-----Test Loss")
    loss = PINN.loss(xy_train_Nu,u_train_Nu)# use mean squared error
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # er = PINN.test()
    lossADAM[i] = loss.detach().cpu().numpy()
    # errorADAM[i] = er.detach().cpu().numpy()
    if i%(steps/10)==0:
        # error_vec = PINN.test()
        print(i,loss.detach().cpu().numpy())
      

elapsed_A = time.time() - start_time_A                
u_predict1=PINN(X_test)
# e_adam_rel2n = torch.linalg.norm((U_test-u_predict1),2)/torch.linalg.norm(U_test,2)#torch.mean(torch.square(U_test - u_predict1)) / Nu#torch.linalg.norm((U_test-u_predict1),2)/Nu    
# e_adam_mae = torch.max(torch.abs(U_test - u_predict1))
print('----------------------------------------------')  
# print('ADAM reL2norm Error: %.2e'  % (e_adam_rel2n))
# print('ADAM MAE Error: %.2e'  % (e_adam_mae))
print('Training time Adam: %.2f' % (elapsed_A))
print('----------------------------------------------')  



##############################
#########LBFGS################
start_time_L = time.time()

'L-BFGS Optimizer'
optimizer = torch.optim.LBFGS(PINN.parameters(), lr2, 
                              max_iter = steps2, 
                              max_eval = None, 
                              tolerance_grad = 1e-11, 
                              tolerance_change = 1e-11, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')

optimizer.step(PINN.closure)
elapsed_L = time.time() - start_time_L       
# Plot loss
if steps != 0:
    plt.plot(np.array(range(steps)) / 1000, lossADAM, linewidth=0.5,\
              linestyle=':', color='b', label='SQR-ResNet')
    # plt.plot(np.array(range(steps)) / 1000, errorADAM, linewidth=2,\
    #          linestyle='-', color='g', label='Plain NN')
    plt.xlim(0, np.ceil(steps / 1000))  # Set x-axis limits to start from zero
if steps2 != 0:
    if steps == 0:
        i = 0
    plt.plot(np.array(range(i, i + len(lossLBFGS))) / 1000,\
              lossLBFGS, linewidth=2, linestyle=':', color='b', label='SQR-ResNet')
    # plt.plot(np.array(range(i, i + len(errorLBFGS))) / 1000,\
    #          errorLBFGS, linewidth=2, linestyle='-', color='g', label='Plain NN')
    plt.xlim(0, np.ceil(steps2 / 1000))  # Set x-axis limits to start from zero

plt.xlabel('Epoch (x10^3)', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Display only integer x-axis tick labels
plt.grid(axis='y', linestyle='--')  # Display horizontal grid lines as dashed lines
plt.grid(axis='x', which='both', linestyle='-', linewidth=0)  # Remove vertical grid lines
# plt.title('Loss with skip connection and Train dSC', fontsize=16)
plt.yscale('log')
plt.legend(fontsize=18)
plt.title('Hand, n={}'.format(ns), fontsize=16)
plt.tight_layout()
plt.show()


u_predict2 = PINN(X_test)
# e_lbfgs_rel2n = torch.linalg.norm((U_test-u_predict2),2)/torch.linalg.norm(U_test,2)  # Relative L2 Norm of the error (Vector)
# e_lbfgs_mae =  torch.max(torch.abs(U_test - u_predict2))  # MAE
print('----------------------------------------------')
# print('LBFGS reL2norm Error: %.2e' % (e_lbfgs_rel2n))
# print('LBFGS MAE Error: %.2e' % (e_lbfgs_mae))
print('Training time LBFGS: %.2f' % (elapsed_L))
print('----------------------------------------------')


#############################################################
##########################################################################
##########################################################################

# pfa = pfa.detach().numpy()
# pfl = pfl.detach().numpy()

# with open("pf.txt", "w") as file:
#     for row in pfa:
#         file.write(" ".join(str(item) for item in row) + "\n")

# with open("pf.txt", "w") as file:
#     for row in pfl:
#         file.write(" ".join(str(item) for item in row) + "\n")
        
        
# Save the tensors in a MATLAB data file (.mat)
mat_data = {
    'pfa': u_predict2.detach().cpu().numpy(),
}

scipy.io.savemat('pf_data.mat', mat_data)
