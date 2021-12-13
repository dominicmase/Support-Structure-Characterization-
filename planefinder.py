# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:00:50 2021

@author: dmase
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.linalg import lstsq
from mpl_toolkits.mplot3d import Axes3D
from iminuit import Minuit

# we also need a cost function to fit and import the LeastSquares function
from iminuit.cost import LeastSquares

"""
Load data from csv and assign x, y, z variables.
"""

#data = (np.array(pd.read_csv('OuterBand_Vertical_test10812_p154.csv', header = 20)))
data = (np.array(pd.read_csv('InnerBand_Vertical_test10812_p88.csv', header = 20)))
x_data = np.array([data[i*3,3] for i in range(0,int(len(data)/3))])
y_data = np.array([data[(i*3)+1,2] for i in range(0,int(len(data)/3))])
z_data = np.array([data[(i*3)+2,2] for i in range(0,int(len(data)/3))])

"""
Do plane fit
"""
def plane_fit(x_data,y_data):
    tmp_A = []
    tmp_b = []
    for i in range(len(x_data)):
        tmp_A.append([x_data[i], y_data[i], 1])
        tmp_b.append(z_data[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    
    fit, residual, rnk, s = lstsq(A, b)
    return fit, residual, rnk 
def plane_fit2(A,B,C):
    data = (np.array(pd.read_csv('InnerBand_Vertical_test10812_p88.csv', header = 20)))
    x = np.array([data[i*3,3] for i in range(0,int(len(data)/3))])
    y = np.array([data[(i*3)+1,2] for i in range(0,int(len(data)/3))])
    z = np.array([data[(i*3)+2,2] for i in range(0,int(len(data)/3))])
    return sum((A*x + B*y + z)**2 / np.sqrt(A**2 + B**2 + 1))
    

"""
plot plane
"""
fit, residual, rnk = plane_fit(x_data,y_data)
plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(x_data, y_data, z_data, color='b')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
print([((xlim[1]-xlim[0])/2)+xlim[0],((ylim[1]-ylim[0])/2)+ylim[0],0])
center = [((xlim[1]-xlim[0])/2)+xlim[0],((ylim[1]-ylim[0])/2)+ylim[0],((zlim[1]-zlim[0])/2)+zlim[0]]
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
rho = np.sqrt((X-center[0])**2+(Y-center[1])**2)
rho_max = ((xlim[1]-xlim[0])/2 + (xlim[1]-xlim[0])/2)/2 #inscibed circle radius based on the xlim and ylim 
#X,Y,Z = np.where(rho < rho_max, X, center[0]), np.where(rho < rho_max, Y,center[1]),np.where(rho < rho_max, Z,center[2])
ax.plot_surface(X, Y, Z, alpha=.6)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(center[0],center[1],center[2],c = 'r')
plt.show()
n = 5
a1, a2, a3 = np.zeros(n), np.zeros(n), np.zeros(n)
angle = 2*np.pi/n
phi_data = np.arctan2(y_data-center[1],x_data-center[0])+np.pi

Chi = []

for j in range(0,len(a1)):    
    """
    slice data
    """
    step = np.array([[x_data[i],y_data[i],z_data[i]] for i in range(0,len(x_data)) if phi_data[i] > (angle*j) and phi_data[i] < (angle*j)+angle])
    x_step, y_step, z_step = step[:,0], step[:,1], step[:,2]
    """
    fit plane
    """
    fit, residual, rnk = plane_fit(x_step,y_step)
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x_step, y_step, z_step, color='b')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.scatter(x_data, y_data, z_data, color='r',s = 20)
    X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
    ax.plot_surface(X, Y, Z, alpha=.6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(center[0],center[1],center[2],c = 'r')
    a1[j], a2[j], a3[j] = fit[0], fit[1], fit[2]
    magnitude = np.sqrt(a1**2+a2**2)
    mu_norm = a1[j]*x_data+a2[j]*y_data+a3[j]*z_data 
    Chi.append(sum((fit[0]*x_data + fit[1]*y_data + fit[2])**2 / np.sqrt(fit[0]**2 + fit[1]**2 + 1)))
    plt.show()


# least_squares = LeastSquares(x_data, y_data, z_data, plane_fit2)
#m = Minuit(leat_squares, A = fit[0], B = fit[1])  
Chi = np.array(Chi)
plt.scatter(range(0,len(Chi**2)),Chi**2-(np.mean(Chi**2)))
plt.title('Sum of diffrences between points and fit')
plt.xlabel('phi step')
plt.show()