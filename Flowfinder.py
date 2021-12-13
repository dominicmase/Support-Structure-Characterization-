# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:01:26 2021
https://www.extendoffice.com/documents/excel/5537-excel-batch-convert-to-csv.html 
@author: dmase
"""
import numpy as np
import image_detection as img
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
file = img.files(r"C:\Users\dmase\Desktop\Rafael lab\2021\Thermal Data 9-8-2021 (csv)\*.csv")
circle_info = img.circle_finder(file)
slices = np.array(img.iso(circle_info,file))
for x in range(len(slices)):
    slices[x] = np.array(slices[x])
    plt.plot(slices[x][:,0],slices[x][:,1])

plt.show()
frame_data = []
for y in range(len(file)-100): #number is how many frames to the right to exclude
    plot_data = (img.data_gen(file[y],slices))
    frame_data.append(plot_data)
erf_param = []
m_plot = []
m_err_plot = []
for y in range(len(slices)):
    print(y)
    erf_param = []
    t = [x for x in range(len(slices[0]))] #this should be the x axis which is the frame number
    print('t',t)
    rho_index = int(len(t)/2) #takes (roughly) the middle of the ring and uses the error function of that (hopefully the fit for this pixel worked) 
    y_data = (np.array(frame_data)[:,y][:,rho_index]) #used for fit 
    plt.scatter(np.linspace(0,len(y_data),len(y_data)),y_data,s = 10)
    plt.plot(img.erf_fit(y_data)[0],img.erf_fit(y_data)[1])
    plt.xlabel('frame')
    plt.ylabel('degrees C')
    plt.title('Temp at Center of Annulus Over Time')
#    plt.xlim(0,20)
    plt.show()
    plt.scatter(img.erf_fit(y_data)[0],(img.erf_fit(y_data)[1])-y_data, s = 10)
    plt.title('Error Function Residuals')
    plt.xlabel('frame')
    plt.ylabel('degrees C')
    plt.ylim(-1,1)
    plt.show()
    m = (img.erf_fit(y_data)[2])[2] #finds m at middle pixel
    m_err = (img.erf_fit(y_data)[3])[2]
    frame_index = int(np.round(m))
    rho_data = (np.array(frame_data)[:,y][frame_index,:])
    plt.scatter(np.linspace(0,len(rho_data),len(rho_data)),rho_data,s = 10)
    gaus_fit = img.gaussian_fit(range(len(rho_data)),rho_data)
    plt.plot(gaus_fit[0],gaus_fit[1])
    plt.title('t = m')
    plt.xlabel('frame')
    plt.ylabel('degrees C')    
    plt.show()
    if frame_index > 0:        
        rho_data = (np.array(frame_data)[:,y][frame_index,:])
        plt.scatter(np.linspace(0,len(rho_data),len(rho_data)),rho_data,s = 10)
        plt.title('phi step at a single frame')
        plt.xlabel('rho (pixel units)')
        plt.ylabel('degrees C')
        gaus_fit = img.gaussian_fit(range(len(rho_data)),rho_data)
        gaus_res = (rho_data-gaus_fit[1])
        gaussmult = img.gaussmult(range(len(rho_data)),rho_data)
        gaussmult_res = (rho_data-gaussmult[1])
        gaussadd = img.gaussadd(range(len(rho_data)),rho_data)
        gaussadd_res = (rho_data-gaussadd[1])
        plt.plot(gaus_fit[0],gaus_fit[1])
        plt.plot(gaussmult[0],gaussmult[1])
        plt.plot(gaussadd[0],gaussadd[1])
        plt.ylim(20,25)
        plt.show()
        plt.scatter(gaus_fit[0],gaus_res,s = 10)
        plt.ylim(-2,2)
        plt.title('Gaussian Residuals')
        plt.xlabel('rho (pixel units)')
        plt.ylabel('degrees C')
        plt.show()
        plt.scatter(gaussmult[0],gaussmult_res,s = 10)
        plt.title('Gaussian Product Residuals')
        plt.xlabel('rho (pixel units)')
        plt.ylabel('degrees C')
        plt.ylim(-2,2)
        plt.show()
        plt.scatter(gaussadd[0],gaussadd_res,s = 10)
        plt.title('Gaussian Sum Residuals')
        plt.xlabel('rho (pixel units)')
        plt.ylabel('degrees C')
        plt.ylim(-2,2)
        plt.show()
    else:
        print("fit failed m is negitive")
    if m > 0:
        m_plot.append([y+1,m])
        m_err_plot.append([y+1,m_err])
plt.show()

def poly_func1(x, a, b,):
    return(a*x+b)

m_plot = np.array(m_plot)
m_err_plot = np.array(m_err_plot)
plt.title('m v phi')
plt.xlabel('phi step')
plt.ylabel('m')
popt, pcov = curve_fit(poly_func1,m_plot[:,0],m_plot[:,1])
plt.errorbar(m_plot[:,0],m_plot[:,1],m_err_plot[:,1], capsize=10, ls = 'none')
plt.scatter(m_plot[:,0],m_plot[:,1], s = 10)
plt.plot(m_plot[:,0],poly_func1(m_plot[:,0],*popt))
err_res = (poly_func1(m_plot[:,0],*popt))-m_plot[:,1]
print('m',*popt)
print('err',*pcov)
plt.show()

print(popt)
plt.show() 
for i in range(len(slices)): 
    slope_frame = int(round(popt[1]*i+popt[0])) #a(phi)+b
    slope_plot = (np.array(frame_data)[:,i][slope_frame,:])
    gaus_fit = img.gaussian_fit(np.linspace(0,len(slope_plot),len(slope_plot)),slope_plot)
    gausmult = img.gaussmult(np.linspace(0,len(slope_plot),len(slope_plot)),slope_plot)
    gausadd = img.gaussadd(np.linspace(0,len(slope_plot),len(slope_plot)),slope_plot)
#    plt.scatter(np.linspace(0,len(slope_plot),len(slope_plot)),slope_plot,s = 10)
    if all(gaus_fit[1]) != 0:
        plt.scatter(np.linspace(0,len(slope_plot),len(slope_plot)),slope_plot,s=10)
        plt.title('t = m')
        plt.xlabel('frame')
        plt.ylabel('degrees C') 
#    if all(gausmult[1]) != 0:
#        plt.plot(np.linspace(0,len(slope_plot),len(slope_plot)),gausmult[1])
#    if all(gausadd[1]) !=0:
#        plt.plot(np.linspace(0,len(slope_plot),len(slope_plot)),gausadd[1])
    gaus_res = (slope_plot-gaus_fit[1])
    gausmult_res = ((slope_plot-gausmult[1]))
    gaussadd_res = ((slope_plot-gausadd[1]))
#    plt.ylim(22.7,24.2)
#    plt.title('t = a*phi+b')
#    plt.xlabel('frame')
#    plt.ylabel('degrees C')
#    plt.show()
#    plt.scatter(np.linspace(0,len(slope_plot),len(slope_plot)),gaus_res,s=10)
#    plt.title('Gaussian Residuals')
#    plt.xlabel('rho (pixel units)')
#    plt.ylabel('degrees C')
#    plt.ylim(-1,1)
#    plt.show()
#    plt.scatter(np.linspace(0,len(slope_plot),len(slope_plot)),gausmult_res,s=10)
#    plt.title('Gaussian Product Residuals')
#    plt.xlabel('rho (pixel units)')
#    plt.ylabel('degrees C')
#    plt.ylim(-1,1)
#    plt.show()
#    plt.show()
#    plt.scatter(np.linspace(0,len(slope_plot),len(slope_plot)),gaussadd_res,s=10)
#    plt.title('Gaussian Sum Residuals ')
#    plt.xlabel('rho (pixel units)')
#    plt.ylabel('degrees C')
#    plt.ylim(-1,1)
#    plt.show()
#    