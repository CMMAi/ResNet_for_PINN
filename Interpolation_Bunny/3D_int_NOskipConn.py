# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:12:49 2023

@author: MI
"""

import matplotlib.ticker as ticker
# import plots
import RHS3D
import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from matplotlib import cm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split

import numpy as np
import time
import scipy.io
import warnings
warnings.filterwarnings("ignore")

#Set default dtype to float32
torch.set_default_dtype(torch.float)

#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if device == 'cuda': 
    print(torch.cuda.get_device_name()) 
print(device)

## Adam Optimizer
steps=0
lr=1e-4

## LBFGS Optimizer
steps2=2000
lr2=1e-1



# To generate new data:
x_min=0
x_max=1
t_min=0
t_max=1
CASE = 1
noise = 'nonoise' #.01#
#Nu: Number of training points 
Nu=5000



my_list = [3] + [50] * 20 + [1]
layers = np.array(my_list)

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
            # if i==0:
            #     z0 = z
            # a = (z) + a
            # if self.skip_connection and (i) % 2 == 0:  # Skip connection every 2 layers
            #     a = (z)**2 +  a #(z + a)  # Add skip connectionz + 
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
        err = PINN.test()
        lossLBFGS.append(loss.detach().cpu().numpy())
        errorLBFGS.append(err.detach().cpu().numpy())
        self.iter += 1
        if self.iter % 100 == 0:
            error_vec = PINN.test()
            print(self.iter,loss.detach().cpu().numpy(),error_vec.detach().cpu().numpy())
        return loss        
    'test neural network'
    def test(self):
        # error_vec=PINN.lossBC(X_test,U_test)
        # error_vec = torch.mean((U_test-u_pred)**2)
        u_pred = self.forward(X_test)
        error_vec =  torch.linalg.norm((U_test-u_pred),2)/torch.linalg.norm(U_test,2)#  torch.linalg.norm((U_test-u_pred),2)/Nu  #    # Relative L2 Norm of the error (Vector)
        u_pred = u_pred.cpu().detach().numpy()
        # u_pred = np.reshape(u_pred,(total_points_x,total_points_y),order='F')

        return error_vec



################################################
data = scipy.io.loadmat(r"G:\My Drive\ColabNotebooks\Juan\PINN-Interpolation\simple NN\skipp connection\Adam\3D\Data3D_Bunny2.mat") 
dsite = 10* data['dsites']  
# dsite = dsite[::10, :]  
x = torch.tensor(dsite[:,0].reshape(-1,1))
y = torch.tensor(dsite[:,1].reshape(-1,1))
z = torch.tensor(dsite[:,2].reshape(-1,1))
XY = torch.cat((x, y, z), dim=1)
U  = RHS3D.force(XY[:,0],XY[:,1],XY[:,2],CASE,noise).unsqueeze(1)

#Choose(Nu) points of our available training data:
idx=np.random.choice(XY.shape[0],Nu,replace=False)
xy_train_Nu=XY[idx,:]
u_train_Nu=U[idx,:]
print("Final training data:",xy_train_Nu.shape,u_train_Nu.shape)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# x=torch.linspace(x_min,x_max,total_points_x).view(-1,1)
# y=torch.linspace(t_min,t_max,total_points_y).view(-1,1)
# # Create the mesh 
# X,Y=torch.meshgrid(x.squeeze(1),y.squeeze(1))
# u_real=RHS.force(X,Y,CASE,noise)


# XY = torch.stack([torch.flatten(X), torch.flatten(Y)], dim=1)
# U  = RHS.force(XY[:,0],XY[:,1],CASE,noise).unsqueeze(1)
# # Domain bounds
# lb=XY[0] #first value
# ub=U[-1] #last value 


# #Choose(Nu) points of our available training data:
# idx=np.random.choice(XY.shape[0],Nu,replace=False)
# xy_train_Nu=XY[idx,:]
# u_train_Nu=U[idx,:]
# print("Final training data:",xy_train_Nu.shape,u_train_Nu.shape)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


torch.manual_seed(123)
#Store tensors to GPU
xy_train_Nu=xy_train_Nu.float().to(device)#Training Points (BC)
u_train_Nu=u_train_Nu.float().to(device)#Training Points (BC)

X_test=XY.float().to(device) # the input dataset (complete)
U_test=U.float().to(device) # the real solution 
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
    er = PINN.test()
    lossADAM[i] = loss.detach().cpu().numpy()
    errorADAM[i] = er.detach().cpu().numpy()
    if i%(steps/10)==0:
        error_vec = PINN.test()
        print(i,loss.detach().cpu().numpy(),error_vec.detach().cpu().numpy())
      

elapsed_A = time.time() - start_time_A                
u_predict1=PINN(X_test)
e_adam_rel2n = torch.linalg.norm((U_test-u_predict1),2)/torch.linalg.norm(U_test,2)#torch.mean(torch.square(U_test - u_predict1)) / Nu#torch.linalg.norm((U_test-u_predict1),2)/Nu    
e_adam_mae = torch.max(torch.abs(U_test - u_predict1))
print('----------------------------------------------')  
print('ADAM reL2norm Error: %.2e'  % (e_adam_rel2n))
print('ADAM MAE Error: %.2e'  % (e_adam_mae))
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
    # plt.plot(np.array(range(steps)) / 1000, lossADAM, linewidth=0.5,\
    #          linestyle=':', color='k', label='')
    plt.plot(np.array(range(steps)) / 1000, errorADAM, linewidth=1,\
             linestyle='-', color='k', label='Plain NN')
    plt.xlim(0, np.ceil(steps / 1000))  # Set x-axis limits to start from zero
if steps2 != 0:
    if steps == 0:
        i = 0
        plt.plot(np.array(range(i, i + len(lossLBFGS))) / 1000,\
                  lossLBFGS, linewidth=2, linestyle=':', color='k', label='')
    plt.plot(np.array(range(i, i + len(errorLBFGS))) / 1000,\
             errorLBFGS, linewidth=2, linestyle='-', color='k', label='Plain NN')
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
plt.title('Stanford bunny, n={}'.format(Nu), fontsize=18)
plt.tight_layout()
plt.show()


u_predict2 = PINN(X_test)
e_lbfgs_rel2n = torch.linalg.norm((U_test-u_predict2),2)/torch.linalg.norm(U_test,2)  # Relative L2 Norm of the error (Vector)
e_lbfgs_mae =  torch.max(torch.abs(U_test - u_predict2))  # MAE
print('----------------------------------------------')
print('LBFGS reL2norm Error: %.2e' % (e_lbfgs_rel2n))
print('LBFGS MAE Error: %.2e' % (e_lbfgs_mae))
print('Training time LBFGS: %.2f' % (elapsed_L))
print('----------------------------------------------')

# Calculate the prediction errors
error_adam = np.linalg.norm(U_test.cpu().numpy() - u_predict1.cpu().detach().numpy(), axis=1)
error_lbfgs = np.linalg.norm(U_test.cpu().numpy() - u_predict2.cpu().detach().numpy(), axis=1)


from mpl_toolkits.mplot3d import Axes3D

# ...

# Create a 3D scatter plot for Adam optimization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(XY.cpu().numpy()[:, 0], XY.cpu().numpy()[:, 1],\
                     XY.cpu().numpy()[:, 2], c=error_adam, cmap='brg',\
                         marker='o', s=30)
ax.set_axis_off()
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('z')
colorbar = plt.colorbar(scatter, label='Error')
colorbar.ax.tick_params(labelsize=20)  # Set the font size of colorbar tick labels
colorbar.set_label('Error', fontsize=20)  # Set the font size of colorbar label
plt.title('Plain NN (Adam)', fontsize=20)
ax.view_init(elev=90, azim=-90)  # elev=90 looks from directly above
plt.show()

# Create a 3D scatter plot for LBFGS optimization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(XY.cpu().numpy()[:, 0], XY.cpu().numpy()[:, 1],\
                     XY.cpu().numpy()[:, 2], c=error_lbfgs, cmap='brg',\
                         marker='o', s=30)
ax.set_axis_off()
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.colorbar(scatter, label='Error')
colorbar = plt.colorbar(scatter, label='Error')
colorbar.ax.tick_params(labelsize=20)  # Set the font size of colorbar tick labels
colorbar.set_label('Error', fontsize=20)  # Set the font size of colorbar label
plt.title('Plain NN (L-BFGS-B)', fontsize=20)
ax.view_init(elev=90, azim=-90)  # elev=90 looks from directly above
plt.show()

##################################################################
#####################################################################


###################################################
###################################################

# plot3D_Matrix(X,Y,u_real)

# # #### Plotting the error profile
# error_surface = (U_test - u_predict2).detach().cpu().numpy()
# # Reshape error_surface to match the shapes of arr_x1 and arr_T1
# error_surface = error_surface.reshape(arr_x1.shape)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(arr_x1, arr_T1, error_surface, cmap='viridis')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('Error')
# ax.set_zlim(-0.030, 0.006)  # Set the z-axis limits
# plt.title('Surface of the Error')
# plt.show()
# ##################################
####### Adam plots
####plotting
# x1=X_test[:,0]
# t1=X_test[:,1]
# arr_x1=x1.reshape(shape=[total_points_x,total_points_y]).transpose(1,0).detach().cpu()
# arr_T1=t1.reshape(shape=[total_points_x,total_points_y]).transpose(1,0).detach().cpu()
# arr_y1=u_predict1.reshape(shape=[total_points_x,total_points_y]).transpose(1,0).detach().cpu()
# arr_y_test=U_test.reshape(shape=[total_points_x,total_points_y]).transpose(1,0).detach().cpu()

# plots.plot2D_Matrix(arr_x1,arr_T1,arr_y1)
# plots.plot3D_Matrix(arr_x1,arr_T1,arr_y1)
# plots.plot2D_Error(arr_x1, arr_T1, arr_y_test, arr_y1,U_test , u_predict1)
###################################################
# ####### LBFGS plots
# x1=X_test[:,0]
# t1=X_test[:,1]
# arr_x1=x1.reshape(shape=[total_points_x,total_points_y]).transpose(1,0).detach().cpu()
# arr_T1=t1.reshape(shape=[total_points_x,total_points_y]).transpose(1,0).detach().cpu()
# arr_y2=u_predict2.reshape(shape=[total_points_x,total_points_y]).transpose(1,0).detach().cpu()
# arr_y_test=U_test.reshape(shape=[total_points_x,total_points_y]).transpose(1,0).detach().cpu()

# plots.plot2D_Matrix(arr_x1,arr_T1,arr_y2)
# plt.title('F{}, n={}, Contour Plot, SQR-SkipResNet'.format(CASE,Nu), fontsize=16)
# plots.plot3D_Matrix(arr_x1,arr_T1,arr_y2)
# plt.title('F{}, n={}, Surface Plot, SQR-SkipResNet'.format(CASE,Nu), fontsize=16)
# plots.plot2D_Error(arr_x1, arr_T1, arr_y_test, arr_y2, U_test , u_predict2)
# plt.title('F{}, n={}, Contour Error Plot, SQR-SkipResNet'.format(CASE,Nu), fontsize=16)
