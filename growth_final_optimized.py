import time 
import numpy as np
import scipy.integrate as integrate
from numpy import sqrt, sin, cos, pi
import math 
import pylab as pp
from pymaxent import moments_c,maxent_reconstruct_c0_1,maxent_reconstruct_c1_1,maxent_reconstruct_c0_2,maxent_reconstruct_c1_2,temp1,temp2
from plyer import notification
import matplotlib.pyplot as plt


start = time.time()

'''Αρχικοποίση των μεταβλητών'''


t=np.linspace(0, 60,31)


initial_m=[]
for i in range(4):
    def distr(L,i=i):
        return (L**i)*0.0399*np.exp(-((L-50)**2)/200)
   
    m, err=integrate.quad(distr, 0, np.inf)
    print('m(',i,')=',m)
    initial_m.append(m)


  
''' Επίλυση του συστήματος διαφορικών εξισώσεων με την Maximum Entropy'''
mu=[initial_m[0],initial_m[1],initial_m[2],initial_m[3]]
sol0,lambdas=maxent_reconstruct_c0_1(mu=mu,bnds=[0,200])
temp1 = lambdas

def moments(t,y):
    m0 = y[0]
    m1 = y[1]
    m2 = y[2]
    m3 = y[3]
    Lmean=m1
    σ=(m2-Lmean**2)**(1/2)
    Lmin=Lmean-3*σ
    Lmax=Lmean+4*σ
    bnds=[Lmin,Lmax]
    L=np.linspace(Lmin,Lmax)
    sol, lambdas=maxent_reconstruct_c1_1(mu=[m0,m1,m2,m3],bnds=bnds)

    print (t)
    dm0dt=0
    
    def moment1(L):
        return(sol(L)+0.002*sol(L)*L)
    
    dm1dt, err1 = integrate.quad(moment1,Lmin,Lmax)
    
    
    def moment2(L):
        return(2*L*(sol(L)+0.002*sol(L)*L))
    
    dm2dt, err2=integrate.quad(moment2,Lmin,Lmax)
    
    def moment3(L):
        return(3*L**2*(sol(L)+0.002*sol(L)*L))
    
    dm3dt, err3=integrate.quad(moment3,Lmin,Lmax)
    
    return(dm0dt,dm1dt,dm2dt,dm3dt)

'''Χρήση της BDF, step by step'''
r=integrate.solve_ivp(moments,[0,60],initial_m,method='BDF',jac=None,t_eval=t,rtol=10**(-4),max_step=0.9)


'''Επίλυση συστήματος με τις ροπές'''

def moments1(y,t):
    m0 = y[0]
    m1 = y[1]
    m2 = y[2]
    m3 = y[3] 
    dm0dt=0
    dm1dt=m0+0.002*m1
    dm2dt=2*(m1+0.002*m2)
    dm3dt=3*(m2+0.002*m3)
    return(dm0dt,dm1dt,dm2dt,dm3dt)

r1=integrate.odeint(moments1, initial_m, t)


plt.figure(0)
fig, ax = plt.subplots(2, 2, figsize=[14, 12])
#Change the figure size
plt.figure(figsize=[16, 12])
#plt.suptitle('Evolution of moments', fontsize=19)
plt.subplot(2,2,1)
plt.plot(r.t,r.y[0,:]/initial_m[0],'.r',t,r1[:,0]/initial_m[0],'-b')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('m0(t)/m0(0)',{"fontsize":16})
pp.legend(('Maximum Entropy','Analytical Solution'),loc='lower right',borderaxespad=0.1, fontsize=15)
pp.title('(a)Evolution of zeroth moment',fontsize=15)
plt.subplot(2,2,2)
plt.plot(r.t,r.y[1,:]/initial_m[1],'.r',t,r1[:,1]/initial_m[1],'-b')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('m1(t)/m1(0)',{"fontsize":16})
pp.title('(b)Evolution of first moment',fontsize=15)
pp.legend(('Maximum Entropy','Analytical Solution'),loc='lower right',borderaxespad=0.1, fontsize=15)
plt.subplot(2,2,3)
plt.plot(r.t,r.y[2,:]/initial_m[2],'.r',t,r1[:,2]/initial_m[2],'-b')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('m2(t)/m2(0)',{"fontsize":16})
pp.title('(c)Evolution of second moment',fontsize=15)
pp.legend(('Maximum Entropy','Analytical Solution'),loc='lower right',borderaxespad=0.1, fontsize=15)
plt.subplot(2,2,4)
plt.plot(r.t,r.y[3,:]/initial_m[3],'.r',t,r1[:,3]/initial_m[3],'-b')
pp.xlabel('t(min)',{"fontsize":16})
pp.ylabel('m3(t)/m3(0)',{"fontsize":16})
pp.title('(d)Evolution of third moment',fontsize=15)
pp.legend(('Maximum Entropy','Analytical Solution'),loc='lower right',borderaxespad=0.1, fontsize=15)
plt.show

'''Χρήση της γνωστής αναλυτικής λύσης που δίνει το paper του Falola'''


'''Αρχική τιμή της κατανομής'''
def no(L):
    return(0.0399*np.exp(-((L-50)**2)/200))


'''Ορισμός της γνωστής τελικής κατανομής από το paper'''

def n(L,t):
    return(no((L+500-500*np.exp(0.002*t))*np.exp(-0.002*t))*np.exp(-0.002*t))

end=time.time()

print(end-start,'s')



'''ΚΑΤΑΣΚΕΥΗ ΧΡΟΝΙΚΩΝ ΓΡΑΦΗΜΑΤΩΝ ΤΗΣ ΚΑΤΑΝΟΜΗΣ'''



'''t=10 min'''

Lmean=r.y[1,5]
σ=(r.y[2,5]-Lmean**2)**(1/2)
Lmin=Lmean-3*σ
Lmax=Lmean+4*σ
bnds=[Lmin,Lmax]
L=np.linspace(Lmin,Lmax)

mu10=[r.y[0,5],r.y[1,5],r.y[2,5],r.y[3,5]]
sol10, lambdas10=maxent_reconstruct_c0_1(mu=mu10,bnds=bnds)
pp.figure(0)
pp.plot(L,sol10(L),'.r')
pp.plot(L,n(L,10),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.title('10 minutes')
pp.legend(('Maximum Entropy','Analytical Solution'),loc=0)
pp.show()

'''t=20 min'''

Lmean=r.y[1,10]
σ=(r.y[2,10]-Lmean**2)**(1/2)
Lmin=Lmean-3*σ
Lmax=Lmean+4*σ
bnds=[Lmin,Lmax]
L=np.linspace(Lmin,Lmax)

mu20=[r.y[0,10],r.y[1,10],r.y[2,10],r.y[3,10]]
sol20, lambdas20=maxent_reconstruct_c0_1(mu=mu20,bnds=bnds)
pp.figure(1)
pp.plot(L,sol20(L),'.r')
pp.plot(L,n(L,20),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.title('20 minutes')
pp.legend(('Maximum Entropy','Analytical Solution'),loc=0)
pp.show()

'''t=30min'''

Lmean=r.y[1,15]
σ=(r.y[2,15]-Lmean**2)**(1/2)
Lmin=Lmean-3*σ
Lmax=Lmean+4*σ
bnds=[Lmin,Lmax]
L=np.linspace(Lmin,Lmax)

mu30=[r.y[0,15],r.y[1,15],r.y[2,15],r.y[3,15]]
sol30, lambdas30=maxent_reconstruct_c0_1(mu=mu30,bnds=bnds)
pp.figure(2)
pp.plot(L,sol30(L),'.r')
pp.plot(L,n(L,30),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.title('30 minutes')
pp.legend(('Maximum Entropy','Analytical Solution'),loc=0)
pp.show()

'''t=40min'''



Lmean=r.y[1,20]
σ=(r.y[2,20]-Lmean**2)**(1/2)
Lmin=Lmean-3*σ
Lmax=Lmean+4*σ
bnds=[Lmin,Lmax]
L=np.linspace(Lmin,Lmax)


mu40=[r.y[0,20],r.y[1,20],r.y[2,20],r.y[3,20]]
sol40, lambdas40=maxent_reconstruct_c0_1(mu=mu40,bnds=bnds)
pp.figure(3)
pp.plot(L,sol40(L),'.r')
pp.plot(L,n(L,40),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.title('40 minutes')
pp.legend(('Maximum Entropy','Analytical Solution'),loc=0)
pp.show()


'''t=50min'''

Lmean=r.y[1,25]
σ=(r.y[2,25]-Lmean**2)**(1/2)
Lmin=Lmean-3*σ
Lmax=Lmean+4*σ
bnds=[Lmin,Lmax]
L=np.linspace(Lmin,Lmax)

mu50=[r.y[0,25],r.y[1,25],r.y[2,25],r.y[3,25]]
sol50, lambdas50=maxent_reconstruct_c0_1(mu=mu50,bnds=bnds)
pp.figure(4)
pp.plot(L,sol50(L),'.r')
pp.plot(L,n(L,50),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.title('50 minutes')
pp.legend(('Maximum Entropy','Analytical Solution'),loc=0)
pp.show()

'''t=60 min'''
Lmean=r.y[1,30]
σ=(r.y[2,30]-Lmean**2)**(1/2)
Lmin=Lmean-3*σ
Lmax=Lmean+4*σ
bnds=[Lmin,Lmax]
L=np.linspace(Lmin,Lmax)

mu60=[r.y[0,30],r.y[1,30],r.y[2,30],r.y[3,30]]
sol60, lambdas60=maxent_reconstruct_c0_1(mu=mu60,bnds=bnds)
pp.figure(5)
pp.plot(L,sol60(L),'.r')
pp.plot(L,n(L,60),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.title('60 minutes')
pp.legend(('Maximum Entropy','Analytical Solution'),loc=0)
pp.show()

plt.figure(1)
fig, ax = plt.subplots(2, 2, figsize=[14, 12])
#Change the figure size
plt.figure(figsize=[16, 12])
#plt.suptitle('Evolution of moments', fontsize=19)
plt.subplot(2,2,1)
plt.plot(L,sol10(L),'.r',L,n(L,10),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.legend(('Maximum Entropy','Analytical Solution'),loc='upper right',borderaxespad=0.1, fontsize=12)
pp.title('(a) t=10 min',fontsize=15)
plt.subplot(2,2,2)
plt.plot(L,sol20(L),'.r',L,n(L,20),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.legend(('Maximum Entropy','Analytical Solution'),loc='upper right',borderaxespad=0.1, fontsize=12)
pp.title('(b) t=20 min',fontsize=15)
plt.subplot(2,2,3)
plt.plot(L,sol40(L),'.r',L,n(L,40),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.legend(('Maximum Entropy','Analytical Solution'),loc='upper right',borderaxespad=0.1, fontsize=12)
pp.title('(c) t=40 min',fontsize=15)
plt.subplot(2,2,4)
plt.plot(L,sol60(L),'.r',L,n(L,60),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.legend(('Maximum Entropy','Analytical Solution'),loc='upper left',borderaxespad=0.1, fontsize=12)
pp.title('(d) t=60 min',fontsize=15)
plt.show

'''TRAVELLING WAVE'''


'''t=10 min'''


L=np.linspace(20,170)
pp.figure(0)
pp.plot(L,sol10(L),'.r')
pp.plot(L,n(L,10),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.title('10 minutes')
pp.legend(('Maximum Entropy','Analytical Solution'),loc=0)
pp.show()

'''t=20 min'''



pp.figure(1)
pp.plot(L,sol20(L),'.r')
pp.plot(L,n(L,20),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.title('20 minutes')
pp.legend(('Maximum Entropy','Analytical Solution'),loc=0)
pp.show()

'''t=30min'''


pp.figure(2)
pp.plot(L,sol30(L),'.r')
pp.plot(L,n(L,30),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.title('30 minutes')
pp.legend(('Maximum Entropy','Analytical Solution'),loc=0)
pp.show()

'''t=40min'''


pp.figure(3)
pp.plot(L,sol40(L),'.r')
pp.plot(L,n(L,40),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.title('40 minutes')
pp.legend(('Maximum Entropy','Analytical Solution'),loc=0)
pp.show()


'''t=50min'''


pp.figure(4)
pp.plot(L,sol50(L),'.r')
pp.plot(L,n(L,50),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.title('50 minutes')
pp.legend(('Maximum Entropy','Analytical Solution'),loc=0)
pp.show()

'''t=60 min'''


pp.figure(5)
pp.plot(L,sol60(L),'.r')
pp.plot(L,n(L,60),'-b')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.title('60 minutes')
pp.legend(('Maximum Entropy','Analytical Solution'),loc=0)
pp.show()

'''Ολικό γράφημα του travelling wave '''

pp.figure(6)
pp.plot(L,sol10(L),L,sol20(L),L,sol30(L),L,sol40(L),L,sol50(L),L,sol60(L))
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.legend(('10 minutes','20 minutes','30 minutes','40 minutes','50 minutes','60 minutes'),loc=0)
pp.show 



message= 'Script has finished running'
notification.notify(message= message,
                    app_icon = None,
                    timeout= 10,
                    toast=False)
