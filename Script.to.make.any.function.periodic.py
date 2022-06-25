# -*- coding: utf-8 -*-
"""
Created on Tue May 24 20:59:27 2022

@author: Manos
"""

'''a(L)=L^3, variable bound integral'''

import time 
import numpy as np
from scipy.integrate import quad,solve_ivp
from numpy import sqrt, pi
import pylab as pp
from pymaxent import moments_c,maxent_reconstruct_c0_1,maxent_reconstruct_c1_1,maxent_reconstruct_c0_2,maxent_reconstruct_c1_2,temp1,temp2
from plyer import notification

'''Temperature cycle'''
Tmax = 308
Tmin = 298
Tr = 273
t1 = 600
t2 = t1+600
t3 = t2+1800
t4 = t3+600
tr = 3600

def theta0(tau):
    if 0<tau<=t1/tr: return (Tmax-Tmin)/Tr*(tr/t1)*tau + Tmin/Tr 
    if t1/tr<tau<=t2/tr: return Tmax/Tr
    if t2/tr<tau<=t3/tr: return -(Tmax-Tmin)/Tr*(tr*tau-t2)/(t3-t2) + Tmax/Tr
    else: return Tmin/Tr
    #if t3/tr<t<=t4/tr: return Tmin/Tr

def theta(tau): return theta0(tau)*Tr

# Function that will convert temperature defined in a given range '[a,b]' to a periodic function of period 'b-a' 
def periodicT(a,b,T,t):
    if t>=a and t<=b :
        return T(t)
    elif t>b:
        t_new=t-(b-a)
        return periodicT(a,b,T,t_new)
    elif t<(a):
        t_new=t+(b-a)
        return periodicT(a,b,t,t_new)
    
def T(tau): return periodicf(0,1,theta,tau)

resolution = 1000
tau = np.linspace(0,1,resolution)

temp=[]
for i in range(resolution):
    tempi = T(tau[i])
    temp.append(tempi)
    
pp.figure(0)
pp.plot(tau,temp)
pp.title('Temperature cycle')
pp.xlabel('t (-)',{"fontsize":16})
pp.ylabel('T(K)',{"fontsize":16})
pp.grid()
pp.show()
