# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:22:10 2019

@author: dmase
"""
import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob #library assositated with file identification
import os  #''
from scipy.optimize import curve_fit
from scipy.stats import crystalball
import scipy as sp
import re
import csv
def rec_finder(file): #img for now but all_files later
    img_color = cv2.imread(file)
    plt.imshow(img_color)
    plt.imshow()
    img = cv2.medianBlur(img_color,5)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 40, 200, cv2.THRESH_BINARY)[1]
    img = cv2.Canny(img, 100, 200)
    plt.imshow(img)
    plt.show()
    conts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(img_color, conts, -1, (-1, 255, 0), 3)
    plt.imshow(img_color)
    plt.show()
    c = max(conts, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    rec = cv2.rectangle(img_color,(x,y),(x+w,y+h),(0,255,0),2)
    plt.imshow(rec)
def circle_finder(all_files):
    print(all_files[-1])
    data = genfromtxt(all_files[-1], delimiter=',',skip_header = 5) #should open last file because it will have the highest contrast
    print(data)
    data = np.delete(data,0,1)
    data = np.delete(data,-1,1)
#    with open(all_files[-1], 'r', encoding='utf16') as csvf: #enables reading the utf16 format 
#        for _ in range(5):
#            next(csvf)
#        for line in csv.reader(csvf):
#            print(line)
#            data.append([float(x) for x in line])

            
#            data.append(line[0].split(","))
#            print(data)
    print(np.shape(data))
#    print(data)
    data = np.delete(data,0,1)
    data = np.delete(data,-1,1)
    plt.imsave('test.png',data)
    plt.imshow(data, cmap = 'magma')
    img_color = cv2.imread('test.png')  #test.png is created in last file function
    img = cv2.medianBlur(img_color,5) #blurs image to make processing easier
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converts to grayscale
    img = np.invert(img)
    cv2.imshow('image',img) #opens image
    cv2.waitKey(5000) #waits for image to render
    cv2.destroyAllWindows() #closes image
    rows = img.shape[0]
    cols = img.shape[1]
    
    def f1(circles):
        
        if circles is not None: 
#            print('found')
            circles = np.uint16(np.around(circles))
            
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(img, center, radius, (255, 0, 255), 1)
        
        else: print('not found')
#        cv2.imshow("detected circles", img)
#        cv2.imwrite('detected circles.png', img) 
#        cv2.waitKey(1000)
        return(radius,center)
    
    def f2(circles,r):
        
        if circles is not None: 
#            print('found')
            circles = np.uint16(np.around(circles))
            
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = r
                cv2.circle(img, center, radius, (255, 0, 255), 1)
        # Simple binary threshold
        
        else: print('not found')
 #       cv2.imshow("detected circles", img) #visual (if not working uncomment)
 #       cv2.imwrite('detected circles.png', img) 
        cv2.waitKey(1000)
    
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,1, rows/8,
                               param1 = 140, param2 = 20, minRadius = 60, maxRadius = 130) #parameters for inner circle 
#    print(circles)
    height,width = img.shape
    radius , center = f1(circles)[0], f1(circles)[1]  #innermost circle
#    print(center)
    r_offset1 = radius + int(radius/1.69) #outer inner circle radius is a multiple of inner radius
    r_offset2 = radius + int(radius*1.2) #inner outer cirlce
    r_offset3 = radius + int(radius*2.3) #outermost circle
    f2(circles,r_offset1)
    f2(circles,r_offset2)
    f2(circles,r_offset3)
#    print(r_offset1, r_offset2, r_offset3) #feed this data into temp reader to form boundary circles
    return(radius,r_offset1,r_offset2,r_offset3,center,rows,cols)
    
def files(path): #opens last file in folder since it will have the highest contrast, then uses that file to id circles
    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    list_of_files = glob.glob(path) 
    all_files = sorted(list_of_files, key=numericalSort)
    print('last_file',all_files[-1])
    return all_files


def iso(circle_info,list_of_files): #isolates important regions of image and plots data. 
    xc = circle_info[4][0]
    yc = circle_info[4][1]
    rows = circle_info[6]
    cols = circle_info[5]
    r, r1 , r2 , r3 = circle_info[0], circle_info[1], circle_info[2], circle_info[3]
#    img = mpimg.imread('test.png') 
#    slice0 = []
#    slice1 = [] 
    theta = (2*np.pi) #how much of the cirlce to slice (axis is rotated by 90 degrees)
#    theta1 = theta+(np.pi)
    n = 10 # slices to make
    thetas = [((x)*(theta/n))-(np.pi/3) for x in range(0,n)] #angle for slices, for outer use -(np.pi/3)
#    thetas1 = []
#    for x in thetas: #keeps thetas within 2pi (python doesnt like it when they become larger)
#        if x > 2*np.pi:
#            thetas1.append(x-(2*np.pi))
#        else:
#            thetas1.append(x)
#    print(thetas1)
#    thetas = np.array(thetas1)
    slices = [x for x in range(0,n)]
#    print(thetas)
    for x in slices:
        slices[x] = []
        for r0 in range(r2,r3):
            x_cord = int(r0*np.sin(thetas[x]))+yc #increments along radius by 1 and then rounds to nearest pixel
            y_cord = int(r0*np.cos(thetas[x]))+xc
#            x_cord1 = int(r0*np.sin(theta))+yc #increments along radius by 1 and then rounds to nearest pixel
#            y_cord1 = int(r0*np.cos(theta))+xc
            #slice1.append([x_cord1,y_cord1])
#            slice0.append([x_cord,y_cord])
#            if x_cord <= cols and y_cord <= rows:  
#                if x_cord > 0 and y_cord > 0 and y_cord < cols: #confines slices to image dimensions 
            slices[x].append([x_cord,y_cord])

#            print(type(slices[x]))
#    print(xc,yc)
    return slices 

def gaussian_fit(x,y):
    def f(x,a1,a2,a3,a4,a5):
        """Model function: Gaussian plus linear background"""     
        return a2+a1*x+a3*np.exp(-(x-a4)**2/(2*a5**2))
#    popt, pcov = curve_fit(f,x,y, sigma = y)
#    data = f(x, *popt)
#    print(*popt)              
    guesses = [-1, 22, -0.1, 25, 10]   # parameter guesses 
#    bound = ([-np.inf,24,-5,45,10],[np.inf,26,-1,65,40])
#    bound = ([-np.inf,-np.inf,-np.inf,45,10],[np.inf,np.inf,np.inf,65,40])
    try:
        (a1,a2,a3,a4,a5),cc = curve_fit(f,x,y,p0=guesses) # do the fit
#        (uaa,ux0,uw,ubb0,ubb1) = 2*np.sqrt(np.diag(cc)) # 2-sigma uncertainties of parameters
        xmod = np.linspace(x[0],x[-1],len(x))       # assumes x's are in order
        ymod = f(xmod,a1,a2,a3,a4,a5)
        data = (xmod,ymod)
    except RuntimeError:
        print('no fit found')
        data = [np.zeros(len(x)),np.zeros(len(y))]
    return data
def gaussmult(x,y):
    def f(x,a1,a2,a3,a4,a5,aa3,aa4,aa5):
        """Model function: 2 Gaussians plus linear background"""     
        return a2+a1*x+a3*((np.exp(-(x-a4)**2/(2*a5**2)))+aa3*(np.exp(-(x-aa4)**2/(2*aa5**2))))
    guesses = [-1, 22, -0.1, 25, 2,-0.1, 12, 5]
    try:
        (a1,a2,a3,a4,a5,aa3,aa4,aa5),cc = curve_fit(f,x,y,p0=guesses)
        xmod = np.linspace(x[0],x[-1],len(x))       # assumes x's are in order
        ymod = f(xmod,a1,a2,a3,a4,a5,aa3,aa4,aa5)
        data = (xmod,ymod)
    except RuntimeError:
        print('no fit found')
        data = [np.zeros(len(x)),np.zeros(len(y))]
    return data
def gaussadd(x,y): #uses boolean arrays 
    def f(x,a1,a2,a3,a4,a5,aa5):
        """Model function: 2 Gaussians plus linear background"""     
        y = np.zeros(len(x))
        y += a2+a1*x+a3*np.exp(-(x-a4)**2/(2*a5**2)) * (x < a4)
        y += a2+a1*x+a3*(np.exp(-(x-a4)**2/(2*aa5**2))) * (x > a4)
        return y
    guesses = [-1,22,-0.1, 25, 10, 10]
    try:
        (a1,a2,a3,a4,a5,aa5),cc = curve_fit(f,x,y,p0=guesses)
        xmod = np.linspace(x[0],x[-1],len(x))       # assumes x's are in order
        ymod = f(xmod,a1,a2,a3,a4,a5,aa5)
        data = (xmod,ymod)
    except RuntimeError:
        print('no fit found')
        data = [np.zeros(len(x)),np.zeros(len(y))]
    return data
#    try:
#        data = (xmod,ymod)
#    except RuntimeError:
#        print('no fit found')
#        data = (0,0)
#    return data

    
    
    

#def crystal_fit(x,y):   
#        def rhs(x, beta, m):
#            return np.exp(-x**2 / 2)
##    popt, pcov = curve_fit(f,x,y, sigma = y)
##    data = f(x, *popt)
##    print(*popt)              
#    guesses = [9.922301063819607, 1,6,30]   # parameter guesses
#    (beta,m,b,c),cc = curve_fit(f,x,y,p0=guesses) # do the fit
#    (ubeta,um,ub,uc) = 2*np.sqrt(np.diag(cc)) # 2-sigma uncertainties of parameters
#    xmod = np.linspace(x[0],x[-1],len(x))       # assumes x's are in order
#    ymod = f(xmod,beta,m,b,c)
#    data = (xmod,ymod)
#    return data
    
def erf_fit(ydata):
    def func(t,a,c0,m,s):#,c1,c2
        return c0+a*sp.special.erf((t-m)/(np.sqrt(2)*s))#+c1*np.exp(-t/c2)
    xdata=np.linspace(0,len(ydata),len(ydata))
    try:
        param_bounds = (([-np.inf,-np.inf,0,-np.inf],[np.inf,np.inf,20,np.inf]))#-np.inf,-np.inf 
        popt,pcov = curve_fit(func,xdata,ydata,p0=[-30,25,10,10], bounds = param_bounds)
        #,3,100
        yfit = func(xdata, *popt)
        perr = np.sqrt(np.diag(pcov))
        
        return(xdata,yfit,popt,perr)
    except RuntimeError:
        yfit = (0,0)
#        print('fit failed')
        return(np.zeros(len(ydata)),np.zeros(len(ydata)),np.zeros(6),[0,0,0,0])

def data_gen(file,slices): #generates a 2d array containing each frame and the data assosicated 
    data_list = [0 for x in range(len(slices))]
    for z in range(0,len(slices)):
        data_list[z] = []
        data = []
        plot_data = []
        data = genfromtxt(file, delimiter=',',skip_header = 5) #should open last file because it will have the highest contrast
        data = np.delete(data,0,1)
        data = np.delete(data,-1,1)
#        plot_data = [data[y[0],y[1]] for y in slices[z]]
        for y in slices[z]:
            if y[0] >= 480 or y[0] < 0 or y[1] >= 640 or y[1] < 0:
                plot_data.append(plot_data[-1]) #if the data is out of range will just append last element of list instead of zero (be mindful of this)
            else: plot_data.append(data[y[0],y[1]])
                
        data_list[z] = np.array(plot_data)   
    return(data_list)