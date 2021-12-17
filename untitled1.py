# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 22:05:44 2021

@author: Manos
"""

import time 
import numpy as np
import scipy.integrate as integrate
from numpy import sqrt, sin, cos, pi
import math 
import pylab as pp
import matplotlib.pyplot as plt
from pymaxent import maxent_reconstruct_c0,maxent_reconstruct_c1,temp
from plyer import notification

start = time.time()


'''Constants'''
a0 = 3.5e-4 #capilary length
lmeand0 = 0.5
lmeanl0 = 0.5
sd0 = 0.025
sl0 = 0.025
m3d0 = 0.062806
ratio = 1.5
m3l0 = m3d0/ratio

Lr = 100e-6 #reference critical size
tr = 3600
Tr = 273
kb = 1e-9
Kbg = kb*tr
q = 10
kg = 1e-4
kd = 2e-4
Eg = 12000
Ed = 12000

'''Temperature cycle'''
T = 308
R = 8.314
Kg = kg*np.exp(-Eg/(R*T))
Kd = kd*np.exp(-Ed/(R*T))


kv = pi/6 
dens = 1300
q0 = 400
q1 = 2500
kr0 = 1e11
Er = 75000
Sd0 = 1
Sl0 = 1
scale = 1e4



'''Αρχικοποίση των μεταβλητών'''
tspan=(0, 100)
teval = np.linspace(tspan[0],tspan[1],1001)


initial_m_d=[]

for i in range(4):
    def f0_d(λ,i=i):
        return( λ**i* ( m3d0*lmeand0*(lmeand0**2+3*sd0**2)/(sqrt(2*pi*sd0)) * np.exp(-(λ-lmeand0)**2/(2*sd0**2)) )*scale )

    m, err=integrate.quad(f0_d, 0, np.inf)
    print('md(',i,')=',m)
    initial_m_d.append(m)


initial_m_l=[]

for i in range(4):
    def f0_l(λ,i=i):
        return( λ**i* ( m3l0*lmeanl0*(lmeanl0**2+3*sl0**2)/(sqrt(2*pi*sl0)) * np.exp(-(λ-lmeanl0)**2/(2*sl0**2)) )*scale )

    m, err=integrate.quad(f0_l, 0, np.inf)
    print('ml(',i,')=',m)
    initial_m_l.append(m)

initial_values = [initial_m_d[0],initial_m_d[1],initial_m_d[2],initial_m_d[3],initial_m_l[0],initial_m_l[1],initial_m_l[2],initial_m_l[3],Sd0,Sl0]




'''Daughter size distribution'''
q = 10
def g(L,λ): return( 3*L**2 * (2*q+1) * ( (2/(λ**3))**(2*q+1) ) * (L**3-(λ**3)/2)**(2*q) )

def g_vera(l,eta): return( 6*l^2/eta^3*(2*q + 1)*(2*l^3/eta^3 - 1)^(2*q) ) 

def cinf_d(T): return( q0*np.exp(-q1/T) )


def cinf_l(T): return( q0*np.exp(-q1/T) )

def γ_d(T): return( cinf_d(T)/(Lr**3*kv*dens) )

def γ_l(T): return( cinf_l(T)/(Lr**3*kv*dens) )


'''Growth Term'''

def σ_d(Sd,T,L): return( Sd-1-a0/(L*Lr*T) )

def σ_l(Sl,T,L): return( Sl-1-a0/(L*Lr*T) )

def G_d(Sd,T,L):
    if σ_d(Sd,T,L)>0: return( tr/Lr * ( kg*np.exp(-Eg/(R*T))*σ_d(Sd,T,L) ) )
    else: return( tr/Lr * ( kd*np.exp(-Ed/(R*T))*σ_d(Sd,T,L) ) )

def G_l(Sl,T,L):
    if σ_l(Sl,T,L)>0: return( tr/Lr * ( kg*np.exp(-Eg/(R*T))*σ_l(Sl,T,L) ) )
    else: return( tr/Lr * ( kd*np.exp(-Ed/(R*T))*σ_l(Sl,T,L) ) )

ee0 = ( initial_m_d[3] - initial_m_l[3] ) / ( initial_m_d[3] + initial_m_l[3] )
ee = []


def moments(t,y):
    m0_d=y[0]
    m1_d=y[1]
    m2_d=y[2]
    m3_d=y[3]
    m0_l=y[4]
    m1_l=y[5]
    m2_l=y[6]
    m3_l=y[7]
    Sd=y[8]
    Sl=y[9]

    print('''current time is''',t,'''min/''',tspan[1],'''min''')
    
    eet1 = (m3_d-m3_l)/(m3_d+m3_l)
    print('''enantiomeric excess is''',eet1)
    ee.append(eet1)
    
    T = 308
    
    Lmean_d=m1_d/m0_d
    σ_d=np.abs(m2_d-Lmean_d**2)**(1/2)
    Lmin_d = 0
    Lmax_d = 7#Lmean_d + 3*σ_d
    print('[Lmin_d,Lmax_d] = ',[Lmin_d,Lmax_d])
    bnds_d=[Lmin_d,Lmax_d] 
    sol_d, lambdas_d= maxent_reconstruct_c1(mu=y[0:4] ,bnds=bnds_d)
    
    Lmean_l=m1_l/m0_l
    σ_l=np.abs(m2_l-Lmean_l**2)**(1/2)
    Lmin_l = 0
    Lmax_l = 7#Lmean_l + 3*σ_l
    print('[Lmin_l,Lmax_l] = ',[Lmin_l,Lmax_l])
    bnds_l=[Lmin_l,Lmax_l]
    
    sol_l, lambdas_l= maxent_reconstruct_c1(mu=y[4:8] ,bnds=bnds_l)
    
    
    
    dm0dt_d = 0
    dm0dt_l = 0
    
    k=1
    def moment1_d(L):
        def σ_d(Sd,T,L): return( Sd-1-a0/(L*Lr*T) )

        if σ_d(Sd,T,L)>0:
            return( (Kg*(Sd-1)*k*L**(k-1) - Kg*a0/(L*Lr*T)*k*L**(k-2))*sol_d(L) )
        else: return( (Kd*(Sd-1)*k*L**(k-1) - Kd*a0/(L*Lr*T)*k*L**(k-2))*sol_d(L)  )
    dm1dt_d = integrate.quad(moment1_d,bnds_d[0],bnds_d[1],epsabs=1e-4,epsrel=1e-4)[0]
    

    def moment1_l(L):
        def σ_l(Sl,T,L): return( Sl-1-a0/(L*Lr*T) )

        if σ_l(Sl,T,L)>0:
            return( (Kg*(Sl-1)*k*L**(k-1) - Kg*a0/(L*Lr*T)*k*L**(k-2))*sol_l(L) )
        else: return( (Kd*(Sl-1)*k*L**(k-1) - Kd*a0/(L*Lr*T)*k*L**(k-2))*sol_l(L)  )
    dm1dt_l = integrate.quad(moment1_l,bnds_l[0],bnds_l[1],epsabs=1e-4,epsrel=1e-4)[0]
    
    k=2
    def moment2_d(L):
        def σ_d(Sd,T,L): return( Sd-1-a0/(L*Lr*T) )

        if σ_d(Sd,T,L)>0:
            return( (Kg*(Sd-1)*k*L**(k-1) - Kg*a0/(L*Lr*T)*k*L**(k-2))*sol_d(L) )
        else: return( (Kd*(Sd-1)*k*L**(k-1) - Kd*a0/(L*Lr*T)*k*L**(k-2))*sol_d(L)  )
    dm2dt_d = integrate.quad(moment2_d,bnds_d[0],bnds_d[1],epsabs=1e-4,epsrel=1e-4)[0]
    

    def moment2_l(L):
        def σ_l(Sl,T,L): return( Sl-1-a0/(L*Lr*T) )

        if σ_l(Sl,T,L)>0:
            return( (Kg*(Sl-1)*k*L**(k-1) - Kg*a0/(L*Lr*T)*k*L**(k-2))*sol_l(L) )
        else: return( (Kd*(Sl-1)*k*L**(k-1) - Kd*a0/(L*Lr*T)*k*L**(k-2))*sol_l(L)  )
    dm2dt_l = integrate.quad(moment2_l,bnds_l[0],bnds_l[1],epsabs=1e-4,epsrel=1e-4)[0]

    k=3
    def moment3_d(L):
        def σ_d(Sd,T,L): return( Sd-1-a0/(L*Lr*T) )

        if σ_d(Sd,T,L)>0:
            return( (Kg*(Sd-1)*k*L**(k-1) - Kg*a0/(L*Lr*T)*k*L**(k-2))*sol_d(L) )
        else: return( (Kd*(Sd-1)*k*L**(k-1) - Kd*a0/(L*Lr*T)*k*L**(k-2))*sol_d(L)  )
    dm3dt_d = integrate.quad(moment3_d,bnds_d[0],bnds_d[1],epsabs=1e-4,epsrel=1e-4)[0]
    

    def moment3_l(L):
        def σ_l(Sl,T,L): return( Sl-1-a0/(L*Lr*T) )

        if σ_l(Sl,T,L)>0:
            return( (Kg*(Sl-1)*k*L**(k-1) - Kg*a0/(L*Lr*T)*k*L**(k-2))*sol_l(L) )
        else: return( (Kd*(Sl-1)*k*L**(k-1) - Kd*a0/(L*Lr*T)*k*L**(k-2))*sol_l(L)  )
    dm3dt_l = integrate.quad(moment3_l,bnds_l[0],bnds_l[1],epsabs=1e-4,epsrel=1e-4)[0]
    
    
    '''Racemization rate'''
    def Rd(Sl,Sd,T):
        return( tr*kr0*np.exp(-Er/(R*T))*(Sl - Sd) )
    def Rl(Sd,Sl,T):
        return( tr*kr0*np.exp(-Er/(R*T))*(Sd - Sl) )
    
    dSddt = -1/(scale*Lr**3*γ_d(T))*dm3dt_d + Rd(Sd,Sl,T)
    dSldt = -1/(scale*Lr**3*γ_l(T))*dm3dt_l + Rl(Sd,Sl,T)
    
    return(dm0dt_d,dm1dt_d,dm2dt_d,dm3dt_d,dm0dt_l,dm1dt_l,dm2dt_l,dm3dt_l,dSddt,dSldt)


r=integrate.solve_ivp(moments ,tspan, initial_values, method='BDF',t_eval=teval, jac=None, max_step = 0.1, rtol=10**(-4))

'''Γράφημα εξέλιξης των ροπών'''

pp.figure(0)
pp.plot(r.t,r.y[0,:],'-')
pp.plot(r.t,r.y[1,:],'-')
pp.plot(r.t,r.y[2,:],'-')
pp.plot(r.t,r.y[3,:],'-')

pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('mκ(t)',{"fontsize":16})
pp.title('Evolution of moments (D)')
#pp.yscale('log')
pp.legend(('k=0','k=1','k=2','k=3'),loc=0)
pp.show()

pp.figure(1)
pp.plot(r.t,r.y[4,:],'-')
pp.plot(r.t,r.y[5,:],'-')
pp.plot(r.t,r.y[6,:],'-')
pp.plot(r.t,r.y[7,:],'-')

pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('mκ(t)',{"fontsize":16})
pp.title('Evolution of moments (L)')
#pp.yscale('log')
pp.legend(('k=0','k=1','k=2','k=3'),loc=0)
pp.show()

pp.figure(2)
pp.plot(r.t,r.y[8,:],'-')
pp.plot(r.t,r.y[9,:],'-')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('Supersaturation',{"fontsize":16})
#pp.xscale('log')
pp.legend(('Sd','Sl'),loc=0)
pp.show()

'''Enantiomeric Excess'''
eet = (r.y[3,:] - r.y[7,:])/(r.y[3,:] + r.y[7,:])
pp.figure(3)
pp.plot(r.t,eet)
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('Enantiomeric Excess',{"fontsize":16})
pp.show()

pp.figure(4)
pp.plot(r.t,r.y[3,:],'-b')
pp.plot(r.t,r.y[7,:],'-r')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('Mass D,L',{"fontsize":16})
pp.legend(('md','ml'),loc=0)
pp.show()

'''t=0'''
k = 0
u = 1
L=np.linspace(k,u,100)

pp.figure(5)

mu0_d=[initial_m_d[0],initial_m_d[1],initial_m_d[2],initial_m_d[3]]
sol0_d, lambdas0_d=maxent_reconstruct_c0(mu=mu0_d,bnds=[k,2])
mu0_l=[initial_m_l[0],initial_m_l[1],initial_m_l[2],initial_m_l[3]]
sol0_l, lambdas0_d=maxent_reconstruct_c0(mu=mu0_l,bnds=[k,2])

pp.plot(L,sol0_d(L),'-b',L,sol0_l(L),'-r')
pp.title('t = 0')
pp.xlabel('λ (-)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})

'''t=30'''
k = 0
u = 2
L=np.linspace(k,u,100)

pp.figure(6)

mu0_d=[r.y[0,1000],r.y[1,1000],r.y[2,1000],r.y[3,1000]]
sol0_d, lambdas0_d=maxent_reconstruct_c0(mu=mu0_d,bnds=[k,2])
mu0_l=[r.y[4,1000],r.y[5,1000],r.y[6,100],r.y[7,1000]]
sol0_l, lambdas0_d=maxent_reconstruct_c0(mu=mu0_l,bnds=[k,2])

pp.plot(L,sol0_d(L),'-b',L,sol0_l(L),'-r')
pp.title('t = 30')
pp.xlabel('λ (-)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})


end=time.time()
print('''total time is ''', (end-start)/60, '''min''')

'''End of script notification'''

message= 'Script has finished running'
notification.notify(message= message,
                    app_icon = None,
                    timeout= 10,
                    toast=False)