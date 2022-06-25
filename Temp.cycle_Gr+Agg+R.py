# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:52:22 2022

@author: Manos
"""

import time 
import numpy as np
from scipy.integrate import quad,solve_ivp
from numpy import sqrt, pi
import pylab as pp
from pymaxent import moments_c,maxent_reconstruct_c0_1,maxent_reconstruct_c1_1,temp1,maxent_reconstruct_c0_2,maxent_reconstruct_c1_2,temp2
from plyer import notification
from scipy.misc import derivative

start = time.time()


'''Constants'''
a0 = 3.5e-4 #capillary length constant [K m]
kv = pi/6 #volume shape factor [-]
dens = 1300 #crystal density [kg m-3]
q0 = 400 #solubility parameter [g g-1]
q1 = 2500 #solubility parameter [K]
kr0 = 1e11 #pre-exponential factor of racemization [s-1]
Er = 75000 #activation energy of racemization [kJ kmol-1]

lmeand0 = 0.5 #initial mean size of population D [-]
lmeanl0 = 0.5 #initial mean size of population L [-]
σd0 = 0.025 #inital standard deviation of population D [-]
σl0 = 0.025 #inital standard deviation of population L [-]
Sd0 = Sl0 = 1 #initial value of supersaturation

m3d0 = 0.062806 #initial distribution of population D parameter [-]
ratio = 1.5 #ratio of m3d0/m3l0 [-]
m3l0 = m3d0/ratio #initial distribution of population L parameter [-]

'''Αδιαστατοποίηση'''
Lr = 100e-6 #reference crystal size [m]
tr = 3600 #reference time (for one cycle) [s]
Tr = 273 #reference temperature [K]
scale = 1e2 #initial distribution scaling parameter [-]

Eg = 12000 #activation energy of growth [kJ kmol-1]
Ed = 12000 #activation energy of dissolution [kJ kmol-1]
R = 8.314 #universal gas constant [kJ K-1 kmol-1]


kb = 0.001 #breakage rate parameter [s-1]
kg = 1e-4 #pre-exponential factor of growth [m s-1]
kd = 2e-4 #pre-exponential factor of dissolution [m s-1]

'''Temperature Profile'''
Tmax = 308 #maximum temperature [K]
Tmin = 298 #minimum temperature [K]
t1 = 600 #end of the heating stage of the cycle [s]
t2 = t1+600 #end of the isothermal stage of the cycle [s]
t3 = t2+1800 #end of the cooling stage of the cycle [s]
t4 = t3+600 #end of the second isothermal stage of the cycle [s]

tol1 = 1e-4 #absolute integration error
tol2 = 1e-3 #relative integration error

'''Αρχικοποίση των μεταβλητών'''
tspan=(0, 20)
step = 0.2
resolution = int(tspan[1]/step)
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

'''Temperature Profile'''

def theta0(tau):
    if 0<tau<=t1/tr: return (Tmax-Tmin)/Tr*(tr/t1)*tau + Tmin/Tr 
    if t1/tr<tau<=t2/tr: return Tmax/Tr
    if t2/tr<tau<=t3/tr: return -(Tmax-Tmin)/Tr*(tr*tau-t2)/(t3-t2) + Tmax/Tr
    else: return Tmin/Tr
    #if t3/tr<t<=t4/tr: return Tmin/Tr

def theta(tau): return theta0(tau)*Tr

# Function that will convert any given function 'f' defined in a given range '[a,b]' to a periodic function of period 'b-a' 
def periodicf(a,b,f,x):
    if x>=a and x<=b :
        return f(x)
    elif x>b:
        x_new=x-(b-a)
        return periodicf(a,b,f,x_new)
    elif x<(a):
        x_new=x+(b-a)
        return periodicf(a,b,f,x_new)
    
def T(tau): return periodicf(0,1,theta,tau)


def Kg(tau): return tr/Lr * kg*np.exp(-Eg/(R*T(tau)))
def Kd(tau): return tr/Lr * kd*np.exp(-Ed/(R*T(tau)))

def cinf_d(tau): return( q0*np.exp(-q1/T(tau)) )
def cinf_l(tau): return( q0*np.exp(-q1/T(tau)) )

def γ_d(tau): return( cinf_d(tau)/(Lr**3*kv*dens) )
def γ_l(tau): return( cinf_l(tau)/(Lr**3*kv*dens) )


'''Growth Term'''
def G_d(Sd,tau,L):
    if L<1e-3: return(0)
    else:
        jd =  Sd-1-a0/(L*Lr*T(tau))
        if jd>0: 
            result = Kg(tau) * jd 
            return result
        else: 
            result = Kd(tau) * jd 
            return result

def G_l(Sl,tau,L):
    if L<1e-3: return(0)
    else:
        jl =  Sl-1-a0/(L*Lr*T(tau))
        if jl>0: 
            result = Kg(tau) * jl 
            return result
        else: 
            result = Kd(tau) * jl 
            return result


'''Racemization rate'''
def Rd(Sl,Sd,tau):
    return( tr*kr0*np.exp(-Er/(R*T(tau)))*(Sl - Sd) )
def Rl(Sd,Sl,tau):
    return( tr*kr0*np.exp(-Er/(R*T(tau)))*(Sd - Sl) )

'''Agglomeration kernel'''
b0=1
def b(L,l):
    return b0


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

    print("\033[1;31m" + '''current time is''',t,'''s/''',tspan[1],'''s''')
    
    eet1 = (m3_d-m3_l)/(m3_d+m3_l)
    print('\033[32m'+'''enantiomeric excess is''',eet1)
    print('\033[39m')
    
    Lmean_d = np.abs(m1_d/m0_d)
    σd = np.sqrt( np.abs(m2_d-2*Lmean_d*m1_d+Lmean_d**2*m0_d)/m0_d )
    Lmean_l=m1_l/m0_l
    σl=np.sqrt( np.abs(m2_l-2*Lmean_l*m1_l+Lmean_l**2*m0_l)/m0_l )
    
    Lmin_d = max(0.25,Lmean_d - 7*σd)
    Lmax_d = min(4.5, Lmean_d + 7*σd)
    Lmin_l = max(0.25,Lmean_l - 7*σl)
    Lmax_l = min(4.5, Lmean_l + 7*σl)

    bnds_d=[Lmin_d,Lmax_d] 
    sol_d, lambdas_d= maxent_reconstruct_c1_1(mu=y[0:4] ,bnds=bnds_d)
    bnds_l=[Lmin_l,Lmax_l]
    sol_l, lambdas_l= maxent_reconstruct_c1_2(mu=y[4:8] ,bnds=bnds_l)
    
    '''Growth functions'''
    def h_d(l,t,k):
        return l**k*sol_d(l)*G_d(Sd,t,l)
    def h_l(l,t,k):
        return l**k*sol_l(l)*G_l(Sl,t,l)
    
    '''Agglomeration functions'''
    def Ad(L):
        def h1(l):
            u = (L**3-l**3)**(1/3)
            if  u<1e-3: temp = 0
            else: temp = b(u,l)/u**2*sol_d(u)*sol_d(l)
            return temp
        result = quad(h1,bnds_d[0],L,epsabs=tol1,epsrel=tol2)[0]
        return result
    def Al(L):
        def h2(l):
            u = (L**3-l**3)**(1/3)
            if  u<1e-3: temp = 0
            else: temp = b(u,l)/u**2*sol_l(u)*sol_l(l)
            return temp
        result = quad(h2,bnds_l[0],L,epsabs=tol1,epsrel=tol2)[0]
        return result
    
    def cd(L):
        if L<1e-3: return 0
        else:
            def h3(l): return b(L,l)*sol_d(l)
            temp = quad(h3,bnds_d[0],bnds_d[1],epsabs=tol1,epsrel=tol2)[0]
        return temp
    def cl(L):
        if L<1e-3: return 0
        else:
            def h4(l): return b(L,l)*sol_l(l)
            temp = quad(h4,bnds_l[0],bnds_l[1],epsabs=tol1,epsrel=tol2)[0]
        return temp
    
    
    
    def momd(L,Sd,T,k):
        return L**k * ( k*L**(-1)*G_d(Sd,t,L)*sol_d(L) + L**2/2*Ad(L) - sol_d(L)*cd(L) )
    
    def moml(L,SL,T,k):
        return L**k * ( k*L**(-1)*G_l(Sl,t,L)*sol_l(L) + L**2/2*Al(L) - sol_l(L)*cl(L) )
    
    k=0
    dm0dt_d = quad(momd,bnds_d[0],bnds_d[1],epsabs=tol1,epsrel=tol2,args=(Sd,t,k))[0] - ( h_d(bnds_d[1],t,k) - h_d(bnds_d[0],t,k) )
    dm0dt_l = quad(moml,bnds_l[0],bnds_l[1],epsabs=tol1,epsrel=tol2,args=(Sd,t,k))[0] - ( h_l(bnds_l[1],t,k) - h_l(bnds_l[0],t,k) )
    k=1
    dm1dt_d = quad(momd,bnds_d[0],bnds_d[1],epsabs=tol1,epsrel=tol2,args=(Sd,t,k))[0] - ( h_d(bnds_d[1],t,k) - h_d(bnds_d[0],t,k) )
    dm1dt_l = quad(moml,bnds_l[0],bnds_l[1],epsabs=tol1,epsrel=tol2,args=(Sd,t,k))[0] - ( h_l(bnds_l[1],t,k) - h_l(bnds_l[0],t,k) )
    k=2
    dm2dt_d = quad(momd,bnds_d[0],bnds_d[1],epsabs=tol1,epsrel=tol2,args=(Sd,t,k))[0] - ( h_d(bnds_d[1],t,k) - h_d(bnds_d[0],t,k) )
    dm2dt_l = quad(moml,bnds_l[0],bnds_l[1],epsabs=tol1,epsrel=tol2,args=(Sd,t,k))[0] - ( h_l(bnds_l[1],t,k) - h_l(bnds_l[0],t,k) )
    k=3
    dm3dt_d = quad(momd,bnds_d[0],bnds_d[1],epsabs=tol1,epsrel=tol2,args=(Sd,t,k))[0] - ( h_d(bnds_d[1],t,k) - h_d(bnds_d[0],t,k) )
    dm3dt_l = quad(moml,bnds_l[0],bnds_l[1],epsabs=tol1,epsrel=tol2,args=(Sd,t,k))[0] - ( h_l(bnds_l[1],t,k) - h_l(bnds_l[0],t,k) )

    dSddt = -1/(scale*Lr**3*γ_d(t))*dm3dt_d + Rd(Sd,Sl,t) - Sd/cinf_d(t)*derivative(cinf_d,t,dx=1e-6)
    dSldt = -1/(scale*Lr**3*γ_l(t))*dm3dt_l + Rl(Sd,Sl,t) - Sl/cinf_l(t)*derivative(cinf_l,t,dx=1e-6)
    
    print('[Lmin_d,Lmax_d] = ',[Lmin_d,Lmax_d])
    print('[Lmin_l,Lmax_l] = ',[Lmin_l,Lmax_l])
    print('dmkdt = \n',[dm0dt_d,dm0dt_l], '\n', [dm1dt_d,dm1dt_l], '\n', [dm2dt_d,dm2dt_l], '\n', [dm3dt_d,dm3dt_l])
    
    return(dm0dt_d,dm1dt_d,dm2dt_d,dm3dt_d,dm0dt_l,dm1dt_l,dm2dt_l,dm3dt_l,dSddt,dSldt)
    

r = solve_ivp(moments ,tspan, initial_values, method='BDF',t_eval=teval, jac=None, rtol=1e-3)


end=time.time()
print('''total time is ''', (end-start)/60, '''min''')

'''End of script notification'''

message= 'Script has finished running'
notification.notify(message= message,
                    app_icon = None,
                    timeout= 10,
                    toast=False)

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

'''Supersaturation'''
pp.figure(4)
pp.plot(r.t,r.y[8,:],'-')
pp.plot(r.t,r.y[9,:],'-')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('Supersaturations',{"fontsize":16})
pp.legend(('Sd','Sl'),loc=0)
pp.show()

eec = [ 0.20000054369455392		,
  0.29183815247871725       ,
  0.3088662398778601        ,
  0.28829752844375095       ,
  0.2647888277570063        ,
  0.26088179668757355       ,
  0.35513361088781026       ,
  0.3597895579199304        ,
  0.3350025545608576        ,
  0.3094142496299399        ,
  0.3039693550572177        ,
  0.39714140619122085       ,
  0.3993950134942731        ,
  0.3748456302796395        ,
  0.34954944426190904       ,
  0.3431776432851193        ,
  0.43321285687039535       ,
  0.43438436542890163       ,
  0.41078725370163366       ,
  0.3858619215565648        ,
  0.37966643039416154       ,
  0.4664533526263868        ,
  0.46751938978813434       ,
  0.4447668494215859        ,
  0.4202238986411776        ,
  0.41285803206216637       ,
  0.5008985396554623        ,
  0.5021845864046364        ,
  0.4799137552485537        ,
  0.45546192357544346       ,
  0.44797239595481775       ,
  0.5314290629446893        ,
  0.5332006304402956        ,
  0.5117432443525309        ,
  0.48758913669049025       ,
  0.4800531055788053        ,
  0.562313427639064         ,
  0.5634676778790746        ,
  0.5424381359652526        ,
  0.5185697473197141        ,
  0.5106342160713223        ,
  0.5892871905660556        ,
  0.5903327754676977        ,
  0.5698114664238134        ,
  0.5462327657582641        ,
  0.5383750938120397        ,
  0.617641095807842         ,
  0.6193783148085186        ,
  0.5990556419128834        ,
  0.5754289340795329        ,
  0.5670122554046769        ,
  0.6439759692919402        ,
  0.6455451770592178        ,
  0.625224833751029         ,
  0.6016674818709039        ,
  0.5930993340172134        ,
  0.6672500336538287        ,
  0.6693327820530919        ,
  0.6491033311601014        ,
  0.625721170373513         ,
  0.6171818074786434        ,
  0.6930751473031858        ,
  0.6956278981633449        ,
  0.6761137124612755        ,
  0.6529623741470058        ,
  0.6443186268150226        ,
  0.7218703741237281        ,
  0.7250215945139937        ,
  0.7059401397699974        ,
  0.6831037273434579        ,
  0.6737723833794764        ,
  0.7521219999220896        ,
  0.7561102990636295        ,
  0.7377103046311052        ,
  0.7153481812890516        ,
  0.7064352284797759        ,
  0.7800485380460467        ,
  0.7840751271997857        ,
  0.7665362257968421        ,
  0.7449899012917098        ,
  0.7358104098763724        ,
  0.8034072328196042        ,
  0.8072596035626723        ,
  0.790786307903457         ,
  0.7704115550167396        ,
  0.7617189290402097        ,
  0.827296464011058         ,
  0.8314954667515968        ,
  0.8163050511778124        ,
  0.7971130888341714        ,
  0.7887310056229774        ,
  0.8514395240858417        ,
  0.8555660145981728        ,
  0.84159398534705          ,
  0.823752230224339         ,
  0.8154285443305941        ,
  0.8725516158607237        ,
  0.87674212856762          ,
  0.8642005906069413        ,
  0.8478912586647829        ,
  0.8403947184296962        ]

'''Enantiomeric Excess'''
eet = (r.y[3,:] - r.y[7,:])/(r.y[3,:] + r.y[7,:])
pp.figure(5)
pp.figure(figsize=[12, 10])
pp.plot(r.t,eet,'-b',r.t,eec,'-r')
pp.xlabel('time',{"fontsize":16})
pp.ylabel('enantiomeric excess',{"fontsize":16})
pp.show()



