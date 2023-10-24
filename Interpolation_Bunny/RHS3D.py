import numpy as np
import matplotlib.pyplot as plt



def force(x,y,z,CASE,noise):
    if CASE == 1: 
        z = 1/3*np.exp(-81/16*((x-0.5)**2+(y-0.5)**2+(z-0.5)**2)) #5th
    if noise != 'nonoise':
        z = z + np.random.randn(x.shape[0])*noise 
    return z
    