# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:03:20 2022

@author: Manos
"""

'''a(L)=L, variable bound integral'''

import time 
import numpy as np
from scipy.integrate import quad,solve_ivp
from numpy import sqrt, pi
import pylab as pp
from pymaxent import moments_c,maxent_reconstruct_c0_1,maxent_reconstruct_c1_1,maxent_reconstruct_c0_2,maxent_reconstruct_c1_2,temp1,temp2
from plyer import notification
from scipy.misc import derivative

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
kb = 0.001
Kbg = kb*tr

kg = 1e-4
kd = 2e-4
Eg = 12000
Ed = 12000

T = 308
R = 8.314
Kg = kg*np.exp(-Eg/(R*T))
Kd = kd*np.exp(-Ed/(R*T))


kv = 0.5236
dens = 1300
q0 = 400
q1 = 2500
kr0 = 1e11
Er = 75000
Sd0 = 1
Sl0 = 1
scale = 1e4

'''Αρχικοποίση των μεταβλητών'''
tspan=(0, 20)
step = 0.5
resolution = int(tspan[1]/step)
teval = np.linspace(tspan[0],tspan[1],resolution+1)


initial_m_d=[]

for i in range(4):
    def f0_d(λ,i=i):
        return( λ**i* ( m3d0*lmeand0*(lmeand0**2+3*sd0**2)/(sqrt(2*pi*sd0)) * np.exp(-(λ-lmeand0)**2/(2*sd0**2)) )*scale )

    m, err=quad(f0_d, 0, np.inf)
    print('md(',i,')=',m)
    initial_m_d.append(m)


initial_m_l=[]

for i in range(4):
    def f0_l(λ,i=i):
        return( λ**i* ( m3l0*lmeanl0*(lmeanl0**2+3*sl0**2)/(sqrt(2*pi*sl0)) * np.exp(-(λ-lmeanl0)**2/(2*sl0**2)) )*scale )

    m, err=quad(f0_l, 0, np.inf)
    print('ml(',i,')=',m)
    initial_m_l.append(m)

initial_values = [initial_m_d[0],initial_m_d[1],initial_m_d[2],initial_m_d[3],initial_m_l[0],initial_m_l[1],initial_m_l[2],initial_m_l[3],Sd0,Sl0]



'''Daughter size distribution'''
q = 2
def g(L,λ): return( 3*L**2 * (2*q+1) * ( (2/(λ**3))**(2*q+1) ) * (L**3-(λ**3)/2)**(2*q) )

'''Breakage kernel'''
def a(l):
    return l

def cinf_d(T): return( q0*np.exp(-q1/T) )
def cinf_l(T): return( q0*np.exp(-q1/T) )

def γ_d(T): return( cinf_d(T)/(Lr**3*kv*dens) )
def γ_l(T): return( cinf_l(T)/(Lr**3*kv*dens) )

'''Racemization rate'''
def Rd(Sl,Sd,T):
    return( tr*kr0*np.exp(-Er/(R*T))*(Sl - Sd) )
def Rl(Sd,Sl,T):
    return( tr*kr0*np.exp(-Er/(R*T))*(Sd - Sl) )


ee0 = ( initial_m_d[3] - initial_m_l[3] ) / ( initial_m_d[3] + initial_m_l[3] )
ee = []

mu0_d=[initial_m_d[0],initial_m_d[1],initial_m_d[2],initial_m_d[3]]
sol0_d,lambdas0_d = maxent_reconstruct_c0_1(mu=mu0_d,bnds=[0,2])
temp1 = lambdas0_d

mu0_l=[initial_m_l[0],initial_m_l[1],initial_m_l[2],initial_m_l[3]]
sol0_l,lambdas0_l = maxent_reconstruct_c0_2(mu=mu0_l,bnds=[0,2])
temp2 = lambdas0_l

'''Integration tolerances'''
tol1 = 1e-4 #absolute error
tol2 = 1e-4 #relative error

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

    print("\033[1;31m" + '''current time is''',t,'''min/''',tspan[1],'''s''')
    
    eet1 = (m3_d-m3_l)/(m3_d+m3_l)
    print('\033[32m'+'''enantiomeric excess is''',eet1)
    print('\033[39m')
    
    Lmean_d = np.abs(m1_d/m0_d)
    σd = np.sqrt( np.abs(m2_d-2*Lmean_d*m1_d+Lmean_d**2*m0_d)/m0_d )
    Lmean_l=m1_l/m0_l
    σl=np.sqrt( np.abs(m2_l-2*Lmean_l*m1_l+Lmean_l**2*m0_l)/m0_l )
    
    Lmin_d = max(0,Lmean_d - 5*σd)
    Lmax_d = Lmean_d + 6*σd
    Lmin_l = max(0,Lmean_l - 5*σl)
    Lmax_l = Lmean_l + 6*σl

    bnds_d=[Lmin_d,Lmax_d] 
    sol_d, lambdas_d= maxent_reconstruct_c1_1(mu=y[0:4] ,bnds=bnds_d)
    bnds_l=[Lmin_l,Lmax_l]
    sol_l, lambdas_l= maxent_reconstruct_c1_2(mu=y[4:8] ,bnds=bnds_l)
    
    
    def fd(L):
        def vd(l): return a(l)*g(L,l)*sol_d(l)
        result = quad(vd,L,bnds_d[1],epsabs=1e-3)[0]
        return result
    
    def fl(L):
        def vl(l): return a(l)*g(L,l)*sol_l(l)
        result = quad(vl,L,bnds_l[1],epsabs=1e-3)[0]
        return result

    def momd(L,k):
        return L**k * ( fd(L)-a(L)*sol_d(L) ) * Kbg
    
    def moml(L,k):
        return L**k * ( fl(L)-a(L)*sol_l(L) ) * Kbg
    
    k=0
    dm0dt_d = quad(momd,bnds_d[0],bnds_d[1],epsabs=tol1,epsrel=tol2,args=(k))[0]
    dm0dt_l = quad(moml,bnds_l[0],bnds_l[1],epsabs=tol1,epsrel=tol2,args=(k))[0]
    k=1
    dm1dt_d = quad(momd,bnds_d[0],bnds_d[1],epsabs=tol1,epsrel=tol2,args=(k))[0]
    dm1dt_l = quad(moml,bnds_l[0],bnds_l[1],epsabs=tol1,epsrel=tol2,args=(k))[0]
    k=2
    dm2dt_d = quad(momd,bnds_d[0],bnds_d[1],epsabs=tol1,epsrel=tol2,args=(k))[0]
    dm2dt_l = quad(moml,bnds_l[0],bnds_l[1],epsabs=tol1,epsrel=tol2,args=(k))[0]
    k=3
    dm3dt_d = quad(momd,bnds_d[0],bnds_d[1],epsabs=tol1,epsrel=tol2,args=(k))[0]
    dm3dt_l = quad(moml,bnds_l[0],bnds_l[1],epsabs=tol1,epsrel=tol2,args=(k))[0]
    
    dSddt = -1/(scale*Lr**3*γ_d(T))*dm3dt_d + Rd(Sd,Sl,T)
    dSldt = -1/(scale*Lr**3*γ_l(T))*dm3dt_l + Rl(Sd,Sl,T)

    print('[Lmin_d,Lmax_d] = ',[Lmin_d,Lmax_d])
    print('[Lmin_l,Lmax_l] = ',[Lmin_l,Lmax_l])
    print('dmkdt = \n',[dm0dt_d,dm0dt_l], '\n', [dm1dt_d,dm1dt_l], '\n', [dm2dt_d,dm2dt_l], '\n', [dm3dt_d,dm3dt_l])
        
    return(dm0dt_d,dm1dt_d,dm2dt_d,dm3dt_d,dm0dt_l,dm1dt_l,dm2dt_l,dm3dt_l,dSddt,dSldt)


r=solve_ivp(moments ,tspan, initial_values, method='BDF',t_eval=teval, jac=None, rtol=1e-4)



'''Γράφημα εξέλιξης των ροπών'''
pp.figure(0)
pp.plot(r.t,r.y[0,:],'-b',r.t,r.y[4,:],'-g')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('m0(t)',{"fontsize":16})
pp.legend(('D','L'),loc=0)
pp.title('Evolution of zeroth moment')
pp.show()

pp.figure(1)
pp.plot(r.t,r.y[1,:],'-b',r.t,r.y[5,:],'-g')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('m1(t)',{"fontsize":16})
pp.legend(('D','L'),loc=0)
pp.title('Evolution of first moment')
pp.show()

pp.figure(2)
pp.plot(r.t,r.y[2,:],'-b',r.t,r.y[6,:],'-g')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('m2(t)',{"fontsize":16})
pp.legend(('D','L'),loc=0)
pp.title('Evolution of second moment')
pp.show()

pp.figure(3)
pp.plot(r.t,r.y[3,:],'-b',r.t,r.y[7,:],'-g')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('m3(t)',{"fontsize":16})
pp.legend(('D','L'),loc=0)
pp.title('Evolution of third moment')
pp.show()

'''Enantiomeric Excess'''
eet = (r.y[3,:] - r.y[7,:])/(r.y[3,:] + r.y[7,:])
pp.figure(5)
pp.plot(r.t,eet)
pp.ylim([0.199, 0.201])
pp.xlabel('time',{"fontsize":16})
pp.ylabel('ee',{"fontsize":16})
pp.title('Evolution of enantiomeric excess',{"fontsize":16})
pp.show()



end=time.time()
print('''total time is ''', (end-start)/60, '''min''')

'''End of script notification'''

message= 'Script has finished running'
notification.notify(message= message,
                    app_icon = None,
                    timeout= 10,
                    toast=False)
