# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:49:21 2021

@author: Manos
"""
import numpy as np
import scipy.integrate as integrate
from numpy import sqrt, sin, cos, pi
import math 
import pylab as pp
from pymaxent import moments_c,maxent_reconstruct_c0_1,maxent_reconstruct_c1_1,temp1
import matplotlib.pyplot as plt
import time 

start = time.time()

'''Αρχικοποίση των μεταβλητών'''
b0=1
tspan=(0, 100)
t=np.linspace(tspan[0],tspan[1],51)

initial_m=[]
for i in range(4):
    def distr(L,i=i):
        return (L**i)*3*L**2*np.exp(-L**3)
   
    m, err=integrate.quad(distr, 0, np.inf)
    print('m(',i,')=',m)
    initial_m.append(m)

''' Επίλυση του συστήματος διαφορικών εξισώσεων με την Maximum Entropy'''
mu=[initial_m[0],initial_m[1],initial_m[2],initial_m[3]]
sol0,lambdas=maxent_reconstruct_c0_1(mu=mu,bnds=[0,100])
temp = lambdas


def moments(t,y):
    m0 = y[0]
    m1 = y[1]
    m2 = y[2]
    m3 = y[3]
    Lmean=m1/m0
    σ=np.abs(m2-Lmean**2)**(1/2)
    Lmin=Lmean-3*σ
    Lmax=Lmean+4*σ
    bnds=[0,Lmax]
    L=np.linspace(Lmin,Lmax)
    λ=np.linspace(Lmin,Lmax)
    
    def b(L,λ):
        b0=1
        b=1
        return(b0*b)
    
    sol, lambdas = maxent_reconstruct_c1_1(mu=[m0,m1,m2,m3],bnds=bnds)
    print('time is',t,'s')
    
    def moment0(L):
        return(-1/2*sol(L)*m0*b(L,λ))
    dm0dt, err0 = integrate.quad(moment0, 0, Lmax)
    def moment1(L,λ):
        return(((L**3+λ**3)**(1/3)/2-L)*sol(L)*sol(λ)*b(L,λ))
    dm1dt, err1=integrate.dblquad(moment1, 0, Lmax, 0, Lmax)   
    
    def moment2(L,λ):
        return(((L**3+λ**3)**(2/3)/2-L**2)*sol(L)*sol(λ)*b(L,λ))
    dm2dt, err2=integrate.dblquad(moment2, 0, Lmax, 0, Lmax)
    
    dm3dt=0
    
    return(dm0dt,dm1dt,dm2dt,dm3dt)


'''Χρήση της BDF, step by step'''

r=integrate.solve_ivp(moments ,tspan, initial_m, method='BDF',t_eval=t, jac=None, rtol=10**(-4))


'''Χρήση γνωστής αναλυτικής σχέσης για τις ροπές'''

def Am0(t):
    return(initial_m[0]*(2/(2+t)))
def Am1(t):
    return(initial_m[1]*(2/(2+t))**(2/3))
def Am2(t):
    return(initial_m[2]*(2/(2+t))**(1/3))


pp.figure(0)

pp.plot(r.t,r.y[0,:]/initial_m[0],'.',t,Am0(t)/initial_m[0],'-')
pp.plot(r.t,r.y[1,:]/initial_m[1],'.',t,Am1(t)/initial_m[1],'-')
pp.plot(r.t,r.y[2,:]/initial_m[2],'.',t,Am2(t)/initial_m[2],'-')
pp.plot(r.t,r.y[3,:]/initial_m[3],'.',[0,100],[1,1],'-')

pp.xlabel('t(sec)',{"fontsize":16})
pp.ylabel('mκ(t)/mκ(0)',{"fontsize":16})
pp.title('Evolution of moments')
pp.yscale('log')
pp.legend(('From MaxEnt','From Analytical'),loc=0)
pp.show()
                               
plt.figure(1)
fig, ax = plt.subplots(2, 2, figsize=[14, 12])
#Change the figure size
plt.figure(figsize=[16, 12])
#plt.suptitle('Evolution of moments', fontsize=19)
plt.subplot(2,2,1)
plt.plot(r.t,r.y[0,:]/initial_m[0],'.r',t,Am0(t)/initial_m[0],'-b')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('m0(t)/m0(0)',{"fontsize":16})
pp.legend(('Maximum Entropy','Analytical Solution'),loc='upper right',borderaxespad=0.1, fontsize=15)
pp.title('(a)Evolution of zeroth moment',fontsize=15)
plt.subplot(2,2,2)
plt.plot(r.t,r.y[1,:]/initial_m[1],'.r',t,Am1(t)/initial_m[1],'-b')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('m1(t)/m1(0)',{"fontsize":16})
pp.title('(b)Evolution of first moment',fontsize=15)
pp.legend(('Maximum Entropy','Analytical Solution'),loc='upper right',borderaxespad=0.1, fontsize=15)
plt.subplot(2,2,3)
plt.plot(r.t,r.y[2,:]/initial_m[2],'.r',t,Am2(t)/initial_m[2],'-b')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('m2(t)/m2(0)',{"fontsize":16})
pp.title('(c)Evolution of second moment',fontsize=15)
pp.legend(('Maximum Entropy','Analytical Solution'),loc='upper right',borderaxespad=0.1, fontsize=15)
plt.subplot(2,2,4)
plt.plot(r.t,r.y[3,:]/initial_m[3],'.r',[0,100],[1,1],'-b')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('m3(t)/m3(0)',{"fontsize":16})
pp.title('(d)Evolution of third moment',fontsize=15)
pp.legend(('Maximum Entropy','Analytical Solution'),loc='upper right',borderaxespad=0.1, fontsize=15)
plt.show


'''Travelling wave'''


'''t=0'''
L=np.linspace(0,2)

Lmean=r.y[1,0]/r.y[0,0]
σ=np.abs(r.y[2,0]-Lmean**2)**(1/2)
Lmin=Lmean-3*σ
Lmax=Lmean+4*σ
bnds=[0,Lmax]


pp.figure(2)
mu0=[initial_m[0],initial_m[1],initial_m[2],initial_m[3]]
sol0, lambdas0=maxent_reconstruct_c0_1(mu=mu0,bnds=bnds)
pp.plot(L,sol0(L))
pp.title('t = 0')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})


'''t=20'''
L=np.linspace(0, 8)

Lmean=r.y[1,10]/r.y[0,10]
σ=np.abs(r.y[2,10]-Lmean**2)**(1/2)
Lmin=Lmean-3*σ
Lmax=Lmean+4*σ
bnds=[0,Lmax]


pp.figure(3)

mu20=[r.y[0,10],r.y[1,10],r.y[2,10],r.y[3,10]]
sol20, lambdas20=maxent_reconstruct_c0_1(mu=mu20,bnds=bnds)
pp.plot(L,sol20(L))
pp.title('t = 20')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})



'''t=40'''

Lmean=r.y[1,20]/r.y[0,20]
σ=np.abs(r.y[2,20]-Lmean**2)**(1/2)
Lmin=Lmean-3*σ
Lmax=Lmean+4*σ
bnds=[0,Lmax]


pp.figure(4)
mu40=[r.y[0,20],r.y[1,20],r.y[2,20],r.y[3,20]]
sol40, lambdas40=maxent_reconstruct_c0_1(mu=mu40,bnds=bnds)
pp.plot(L,sol40(L))
pp.title('t = 40')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})



'''t=60'''

Lmean=r.y[1,30]/r.y[0,30]
σ=np.abs(r.y[2,30]-Lmean**2)**(1/2)
Lmin=Lmean-3*σ
Lmax=Lmean+4*σ
bnds=[0,Lmax]



pp.figure(5)
mu60=[r.y[0,30],r.y[1,30],r.y[2,30],r.y[3,30]]
sol60, lambdas60=maxent_reconstruct_c0_1(mu=mu60,bnds=bnds)
pp.plot(L,sol60(L))
pp.title('t = 60')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})


'''t=80'''

Lmean=r.y[1,40]/r.y[0,40]
σ=np.abs(r.y[2,40]-Lmean**2)**(1/2)
Lmin=Lmean-3*σ
Lmax=Lmean+4*σ
bnds=[0,Lmax]



pp.figure(6)

mu80=[r.y[0,40],r.y[1,40],r.y[2,40],r.y[3,40]]
sol80, lambdas80=maxent_reconstruct_c0_1(mu=mu80,bnds=bnds)
pp.plot(L,sol80(L))
pp.title('t = 80')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})



'''t=100'''

Lmean=r.y[1,50]/r.y[0,50]
σ=np.abs(r.y[2,50]-Lmean**2)**(1/2)
Lmin=Lmean-3*σ
Lmax=Lmean+4*σ
bnds=[0,Lmax]


pp.figure(7)
L=np.linspace(0,8,51)
mu100=[r.y[0,50],r.y[1,50],r.y[2,50],r.y[3,50]]
sol100, lambdas100=maxent_reconstruct_c0_1(mu=mu100,bnds=bnds)
pp.plot(L,sol100(L))
pp.title('t = 100')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})



'''Ολικό γράφημα του travelling wave '''

pp.figure(8)
figsize=[14, 12]
pp.plot(L,sol20(L),L,sol40(L),L,sol60(L),L,sol80(L),L,sol100(L))
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.legend(('20 min','40 min','60 min','80 min','100 min'),loc=0)
pp.show 

end = time.time()
print('Total time =',end-start,'sec')
