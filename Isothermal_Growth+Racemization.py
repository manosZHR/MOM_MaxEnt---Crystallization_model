# -*- coding: utf-8 -*-
"""
Created on Mon May 23 18:26:15 2022

@author: Manos
"""

import time 
import numpy as np
from scipy.integrate import quad,solve_ivp
from numpy import sqrt, pi
import pylab as pp
from pymaxent import moments_c,maxent_reconstruct_c0_1,maxent_reconstruct_c1_1,maxent_reconstruct_c0_2,maxent_reconstruct_c1_2,temp1,temp2
from plyer import notification

start = time.time()

''' Σταθερές Αδιαστατοποίησης'''
Lr = 100e-6 
tr = 3600
Tr = 273
scale = 1e4

'''Constants'''
a0 = 3.5e-4
kv = pi/6 
dens = 1300
q0 = 400
q1 = 2500
kr0 = 1e11
Er = 75000

Sd0 = Sl0 = 1
lmeand0 = lmeanl0 = 0.5
σd0 = σl0 = 0.025

m3d0 = 0.062806
ratio = 1.5
m3l0 = m3d0/ratio

Eg = Ed = 12000
R = 8.314

T = 308

kb = 0.001
kg = 1e-4
kd = 2e-4

Kg = tr/Lr * kg*np.exp(-Eg/(R*T))
Kd = tr/Lr * kd*np.exp(-Ed/(R*T))
Kbg = kb*tr

'''Απόλυτο και σχετικό σφάλμα ολοκλήρωσης'''
tol1 = 1e-5 #absolute error
tol2 = 1e-4 #relative error


'''Αρχικοποίση των μεταβλητών'''
tspan=(0, 30)
resolution = 150
teval = np.linspace(tspan[0],tspan[1],resolution+1)


def f0_d(λ):
    return ( m3d0*lmeand0*(lmeand0**2+3*σd0**2)/(sqrt(2*pi*σd0)) * np.exp(-(λ-lmeand0)**2/(2*σd0**2)) )*scale 
def f0_l(λ):
    return ( m3l0*lmeanl0*(lmeanl0**2+3*σl0**2)/(sqrt(2*pi*σl0)) * np.exp(-(λ-lmeanl0)**2/(2*σl0**2)) )*scale 

initial_m_d = moments_c(f0_d, k=4, bnds=[0, np.inf])
initial_m_l = moments_c(f0_l, k=4, bnds=[0, np.inf])

initial_values = [*initial_m_d, *initial_m_l, Sd0, Sl0]

print("\033[1;36m"+'Initial moments (D)=',initial_m_d,'\nInitial moments (L)=',initial_m_d,'\033[39m')

ee0 = ( initial_m_d[3] - initial_m_l[3] ) / ( initial_m_d[3] + initial_m_l[3] )

def cinf_d(T): return( q0*np.exp(-q1/T) )
def cinf_l(T): return( q0*np.exp(-q1/T) )

def γ_d(T): return( cinf_d(T)/(Lr**3*kv*dens) )
def γ_l(T): return( cinf_l(T)/(Lr**3*kv*dens) )


'''Growth Term'''
def G_d(Sd,T,L):
    if L<1e-3: return(0)
    else:
        jd =  Sd-1-a0/(L*Lr*T)
        if jd>0: 
            result = Kg * jd 
            return result
        else: 
            result = Kd * jd 
            return result

def G_l(Sl,T,L):
    if L<1e-3: return(0)
    else:
        jl =  Sl-1-a0/(L*Lr*T)
        if jl>0: 
            result = Kg * jl 
            return result
        else: 
            result = Kd * jl 
            return result

'''Racemization rate'''
def Rd(Sl,Sd,T):
    return( tr*kr0*np.exp(-Er/(R*T))*(Sl - Sd) )
def Rl(Sd,Sl,T):
    return( tr*kr0*np.exp(-Er/(R*T))*(Sd - Sl) )

'''t=0'''

Lmeand0 = initial_m_d[1]/initial_m_d[0]
sigmad0 = np.sqrt( np.abs(initial_m_d[2]-2*Lmeand0*initial_m_d[1]+lmeand0**2*initial_m_d[0])/initial_m_d[0] )

Lmeanl0 = initial_m_l[1]/initial_m_l[0]
sigmal0 = np.sqrt( np.abs(initial_m_l[2]-2*Lmeanl0*initial_m_l[1]+lmeanl0**2*initial_m_l[0])/initial_m_l[0] )

lmind = Lmeand0 - 12*sigmad0
lmaxd = Lmeand0 + 12*sigmad0

lminl = Lmeanl0 - 12*sigmal0
lmaxl = Lmeanl0 + 12*sigmal0

L=np.linspace(0,1,1000)

pp.figure(6)

mu0_d=[initial_m_d[0],initial_m_d[1],initial_m_d[2],initial_m_d[3]]
sol0_d, lambdas0_d=maxent_reconstruct_c0_1(mu=mu0_d,bnds=[lmind,lmaxd])

mu0_l=[initial_m_l[0],initial_m_l[1],initial_m_l[2],initial_m_l[3]]
sol0_l, lambdas0_l=maxent_reconstruct_c0_2(mu=mu0_l,bnds=[lminl,lmaxl])

pp.plot(L,sol0_d(L),'-b',L,sol0_l(L),'-g')
pp.title('t = 0')
pp.xlabel('λ (-)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})

'''Επίλυση του συστήματος διαφορικών εξισώσεων'''
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
    
    '''Integration bounds'''
    Lmean_d = np.abs(m1_d/m0_d)
    σd = np.sqrt( np.abs(m2_d-2*Lmean_d*m1_d+Lmean_d**2*m0_d)/m0_d )
    Lmean_l=m1_l/m0_l
    σl=np.sqrt( np.abs(m2_l-2*Lmean_l*m1_l+Lmean_l**2*m0_l)/m0_l )
    
    Lmin_d = max(0.25,Lmean_d - 4*σd)
    Lmax_d = Lmean_d + 5*σd
    Lmin_l = max(0.25,Lmean_l - 4*σl)
    Lmax_l = Lmean_l + 5*σl

    bnds_d=[Lmin_d,Lmax_d] 
    sol_d, lambdas_d= maxent_reconstruct_c1_1(mu=y[0:4] ,bnds=bnds_d)
    bnds_l=[Lmin_l,Lmax_l]
    sol_l, lambdas_l= maxent_reconstruct_c1_2(mu=y[4:8] ,bnds=bnds_l)
    
    
    def h_d(l,k):
        return l**k*sol_d(l)*G_d(Sd,T,l)
    def h_l(l,k):
        return l**k*sol_l(l)*G_l(Sl,T,l)
    
    def momd(l,Sd,T,k):
        result = k*l**(k-1)*G_d(Sd,T,l)*sol_d(l)
        return result 
    def moml(l,Sl,T,k):
        result = k*l**(k-1)*G_l(Sl,T,l)*sol_l(l)
        return result 
        
    k = 0
    dm0dt_d = quad(momd,bnds_d[0],bnds_d[1],epsabs=tol1,epsrel=tol2,args=(Sd,T,k))[0] - ( h_d(bnds_d[1],k) - h_d(bnds_d[0],k) )
    dm0dt_l = quad(moml,bnds_l[0],bnds_l[1],epsabs=tol1,epsrel=tol2,args=(Sl,T,k))[0] - ( h_l(bnds_l[1],k) - h_l(bnds_l[0],k) )

    k = 1
    dm1dt_d = quad(momd,bnds_d[0],bnds_d[1],epsabs=tol1,epsrel=tol2,args=(Sd,T,k))[0] - ( h_d(bnds_d[1],k) - h_d(bnds_d[0],k) )
    dm1dt_l = quad(moml,bnds_l[0],bnds_l[1],epsabs=tol1,epsrel=tol2,args=(Sl,T,k))[0] - ( h_l(bnds_l[1],k) - h_l(bnds_l[0],k) )
    
    k=2
    dm2dt_d = quad(momd,bnds_d[0],bnds_d[1],epsabs=tol1,epsrel=tol2,args=(Sd,T,k))[0] - ( h_d(bnds_d[1],k) - h_d(bnds_d[0],k) )
    dm2dt_l = quad(moml,bnds_l[0],bnds_l[1],epsabs=tol1,epsrel=tol2,args=(Sl,T,k))[0] - ( h_l(bnds_l[1],k) - h_l(bnds_l[0],k) )
    
    k=3
    dm3dt_d = quad(momd,bnds_d[0],bnds_d[1],epsabs=tol1,epsrel=tol2,args=(Sd,T,k))[0] - ( h_d(bnds_d[1],k) - h_d(bnds_d[0],k) )
    dm3dt_l = quad(moml,bnds_l[0],bnds_l[1],epsabs=tol1,epsrel=tol2,args=(Sl,T,k))[0] - ( h_l(bnds_l[1],k) - h_l(bnds_l[0],k) )

    dSddt = -1/(scale*Lr**3*γ_d(T))*dm3dt_d + Rd(Sd,Sl,T)
    dSldt = -1/(scale*Lr**3*γ_l(T))*dm3dt_l + Rl(Sd,Sl,T)
    
    print('bnds_d = ',bnds_d)
    print('bnds_l = ',bnds_l)
    print('Lmean d,l = ',Lmean_d,Lmean_l)
    print('[dm0dt_d,dm0dt_l] =',[dm0dt_d,dm0dt_l])
    
    return(dm0dt_d,dm1dt_d,dm2dt_d,dm3dt_d,dm0dt_l,dm1dt_l,dm2dt_l,dm3dt_l,dSddt,dSldt)


r = solve_ivp(moments ,tspan, initial_values, method='BDF', t_eval=teval, jac=None, rtol=1e-3)

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
pp.plot(r.t,r.y[3,:],'-b',r.t,r.y[7,:],'-g',r.t,r.y[3,:]+r.y[7,:],'-r')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('m3(t)',{"fontsize":16})
pp.legend(('D','L'),loc=0)
pp.title('Evolution of third moment')
pp.show()


'''End of script notification'''

message= 'Script has finished running'
notification.notify(message= message,
                    app_icon = None,
                    timeout= 10,
                    toast=False)
end=time.time()
print('''total time is ''', (end-start)/60, '''min''')


