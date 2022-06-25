# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 18:45:42 2022

@author: Manos
"""

''' a(L) = L**3 , q = 0 '''

import numpy as np
import scipy.integrate as integrate
from numpy import sqrt, sin, cos, pi
import math 
import pylab as pp
from pymaxent import moments_c,maxent_reconstruct_c0_1,maxent_reconstruct_c1_1,maxent_reconstruct_c0_2,maxent_reconstruct_c1_2,temp1,temp2
import matplotlib.pyplot as plt
import time


start = time.time()


'''Αρχικοποίση των μεταβλητών'''

tspan=(0, 100)
t=np.linspace(tspan[0],tspan[1],51)

kb = 1

initial_m=[]
for i in range(4):
    def distr(L,i=i):
        return (L**i)*3*L**2*np.exp(-L**3)
   
    m, err=integrate.quad(distr, 0, np.inf)
    print('m(',i,')=',m)
    initial_m.append(m)
    
''' Επίλυση του συστήματος διαφορικών εξισώσεων με την Maximum Entropy'''
mu=[initial_m[0],initial_m[1],initial_m[2],initial_m[3]]
sol0,lambdas=maxent_reconstruct_c0_1(mu=mu,bnds=[0,2])
temp = lambdas

def moments(t,y):
    m0 = y[0]
    m1 = y[1]
    m2 = y[2]
    m3 = y[3]
    
    Lmean=m1/m0
    σ=np.abs(m2-Lmean**2)**(1/2)
    Lmin=Lmean-3*σ
    Lmax=Lmean+3*σ
    bnds=[0,Lmax]
    L=np.linspace(Lmin,Lmax)
    λ=np.linspace(Lmin,Lmax)    
    
    sol, lambdas=maxent_reconstruct_c1_1(mu=[m0,m1,m2,m3],bnds=bnds)
    
    print(t)
    print(bnds)
    print('Lmean=',Lmean)

    def f00(L):
        result = integrate.quad(sol,L,Lmax,epsabs=1e-3)[0]
        return result
    k=0
    def f01(L):
        result = 6*L**(k+2)*f00(L)-L**(k+3)*sol(L)
        return result
    dm0dt = integrate.quad(f01,0,Lmax,epsabs=1e-3)[0]

    k=1
    def f1(L):
        result = 6*L**(k+2)*f00(L)-L**(k+3)*sol(L)
        return result
    
    dm1dt = integrate.quad(f1,0,Lmax,epsabs=1e-3)[0]

    k=2
    def f2(L):
        result = 6*L**(k+2)*f00(L)-L**(k+3)*sol(L)
        return result
    
    dm2dt = integrate.quad(f2,0,Lmax,epsabs=1e-3)[0]
    
    k=3
    def f3(L):
        result = 6*L**(k+2)*f00(L)-L**(k+3)*sol(L)
        return result
    
    dm3dt = integrate.quad(f3,0,Lmax,epsabs=1e-3)[0]


    return(dm0dt,dm1dt,dm2dt,dm3dt)


'''Χρήση της BDF, step by step'''


r=integrate.solve_ivp(moments,tspan,initial_m,method='BDF',t_eval=t, jac=None, rtol=10**(-3))


'''Analytical Solution'''
'''Zero Moment'''

def n0(L,t):
    return(3*L**2*np.exp(-L**3*(1+t))*(1+t)**2)

def An0(t):
   A0,err0=integrate.quad(n0,0,np.inf,args=(t,))
   return A0
final_m0=np.vectorize(An0)


'''Analytical Solution'''
'''First Moment'''
def n1(L,t):
    return(L*3*L**2*np.exp(-L**3*(1+t))*(1+t)**2)

def An1(t):
   A1,err1=integrate.quad(n1,0,np.inf,args=(t,))
   return A1
final_m1=np.vectorize(An1)

'''Analytical Solution'''
'''Second Moment'''

def n2(L,t):
    return(L**2*3*L**2*np.exp(-L**3*(1+t))*(1+t)**2)

def An2(t):
   A2,err2=integrate.quad(n2,0,np.inf,args=(t,))
   return A2
final_m2=np.vectorize(An2)

'''Analytical Solution'''
'''Third Moment'''

def n3(L,t):
    return(L**3*3*L**2*np.exp(-L**3*(1+t))*(1+t)**2)

def An3(t):
   A3,err3=integrate.quad(n3,0,np.inf,args=(t,))
   return A3
final_m3=np.vectorize(An3)


pp.figure(0)
plt.figure(figsize=[8, 6])
pp.plot(r.t,r.y[0,:]/initial_m[0],'.b',t,final_m0(t),'-b')
pp.plot(r.t,r.y[1,:]/initial_m[1],'.r',t,final_m1(t),'-r')
pp.plot(r.t,r.y[2,:]/initial_m[2],'.g',t,final_m2(t),'-g')
pp.plot(r.t,r.y[3,:]/initial_m[3],'.y',t,final_m3(t),'-y')
pp.yscale('log')

pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('mκ(t)/mκ(0)',{"fontsize":16})
pp.title('Evolution of moments')
pp.legend(('Maximum Entropy','Analytical solution'),loc=0)
pp.show()



'''Κατασκευή γραφημάτων της κατανομής για συγκεκριμένες χρονικές στιγμές'''

'''t=0'''


pp.figure(1)


Lmean=r.y[1,0]/r.y[0,0]
σ=np.abs(r.y[2,0]-Lmean**2)**(1/2)
Lmin=Lmean-4*σ
Lmax=Lmean+5*σ
bnds=[Lmin,Lmax]
L=np.linspace(0,2)

mu0=[initial_m[0],initial_m[1],initial_m[2],initial_m[3]]
sol0, lambdas0=maxent_reconstruct_c0_1(mu=mu0, bnds=bnds)
pp.plot(L,sol0(L),'.',L,n0(L,0),'-')
pp.title('t = 0')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})
pp.legend(('From MaxEnt','From Analytical'),loc=0)


'''t=20'''



pp.figure(2)

Lmean=r.y[1,10]/r.y[0,10]
σ=np.abs(r.y[2,10]-Lmean**2)**(1/2)
Lmin=Lmean-4*σ
Lmax=Lmean+5*σ
bnds=[Lmin,Lmax]
L=np.linspace(0,0.8)



mu20=[r.y[0,10],r.y[1,10],r.y[2,10],r.y[3,10]]
sol20, lambdas20=maxent_reconstruct_c0_1(mu=mu20,bnds=bnds)
pp.plot(L,sol20(L),'.',L,n0(L,20),'-')
pp.title('t = 20')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})

pp.legend(('From MaxEnt','From Analytical'),loc=0)


'''t=40'''
pp.figure(3)

Lmean=r.y[1,20]/r.y[0,20]
σ=np.abs(r.y[2,20]-Lmean**2)**(1/2)
Lmin=0
Lmax=Lmean+4*σ
bnds=[Lmin,Lmax]
L=np.linspace(0,0.8)


mu40=[r.y[0,20],r.y[1,20],r.y[2,20],r.y[3,20]]
sol40, lambdas40=maxent_reconstruct_c0_1(mu=mu40,bnds=bnds)
pp.plot(L,sol40(L),'.',L,n0(L,40),'-')
pp.title('t = 40')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})
pp.legend(('From MaxEnt','From Analytical'),loc=0)

'''t=60'''


Lmean=r.y[1,30]/r.y[0,30]
σ=np.abs(r.y[2,30]-Lmean**2)**(1/2)
Lmin=Lmean-4*σ
Lmax=Lmean+5*σ
bnds=[Lmin,Lmax]
L=np.linspace(0,0.7)




pp.figure(4)

mu60=[r.y[0,30],r.y[1,30],r.y[2,30],r.y[3,30]]
sol60, lambdas60=maxent_reconstruct_c0_1(mu=mu60,bnds=bnds)
pp.plot(L,sol60(L),'.',L,n0(L,60),'-')
pp.title('t = 60')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})
pp.legend(('From MaxEnt','From Analytical'),loc=0)

'''t=80'''

Lmean=r.y[1,40]/r.y[0,40]
σ=np.abs(r.y[2,40]-Lmean**2)**(1/2)
Lmin=0#Lmean-σ
Lmax=Lmean+5*σ
bnds=[Lmin,Lmax]
L=np.linspace(0,0.8)


pp.figure(5)

mu80=[r.y[0,40],r.y[1,40],r.y[2,40],r.y[3,40]]
sol80, lambdas80=maxent_reconstruct_c0_1(mu=mu80,bnds=bnds)
pp.plot(L,sol80(L),'.',L,n0(L,80),'-')
pp.title('t = 80')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})
pp.legend(('From MaxEnt','From Analytical'),loc=0)


'''t=100'''

Lmean=r.y[1,50]/r.y[0,50]
σ=np.abs(r.y[2,50]-Lmean**2)**(1/2)
Lmin=Lmean-σ
Lmax=Lmean+5*σ
bnds=[Lmin,Lmax]

pp.figure(6)

mu100=[r.y[0,50],r.y[1,50],r.y[2,50],r.y[3,50]]
sol100, lambdas100=maxent_reconstruct_c0_1(mu=mu100,bnds=bnds)
pp.plot(L,sol100(L),'.',L,n0(L,100),'-')
pp.title('t = 100')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})
pp.legend(('From MaxEnt','From Analytical'),loc=0)



'''Ολικό γράφημα του travelling wave '''

pp.figure(7)
pp.plot(L,sol20(L),L,sol40(L),L,sol60(L),L,sol80(L),L,sol100(L))
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.legend(('20 sec','40 sec','60 sec','80 sec','100 sec'),loc=0)
pp.show 


end = time.time()
print('Total time =',(end-start),'sec')







