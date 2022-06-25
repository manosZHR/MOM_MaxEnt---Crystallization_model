# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""PyMaxEnt.py: Implements a maximum entropy reconstruction of distributions with known moments."""

__author__     = "Tony Saad and Giovanna Ruai"
__copyright__  = "Copyright (c) 2019, Tony Saad"

__credits__    = ["University of Utah Department of Chemical Engineering", "University of Utah UROP office"]
__license__    = "MIT"
__version__    = "1.0.0"
__maintainer__ = "Tony Saad"
__email__      = "tony.saad@chemeng.utah.edu"
__status__     = "Production"

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
temp1=[]
temp2=[]

def moments_c(f, k=0, bnds=[0, np.inf]):
    '''
    Creates "k" moments: μ0, μ1, ..., μ(k-1) for a function "f" on the support given by "bnds".

    Parameters:
        f (function): distribution function **must be in the form of a function**
        k (int): integer number of moments to compute. Will evaluate the first k moments of f, μ0, μ1, ..., μ(k-1)
        bnds (tuple): boundaries for the integration

    Returns:
        moments: an array of moments of length "k"
    
    Example:
        μ = moments(3, f, [-1, 1])    
    '''
    def mom(x, k):
        return x**k*f(x)
    
    moms = np.zeros(k)
    a = bnds[0]
    b = bnds[1]
    for i in range(0,k):
        moms[i] = quad(mom,a,b,args = i)[0]
    return moms


def integrand_1(x, lamb, k=0):
    '''
    Calculates the integrand of the \(k^\mathrm{th}\) moment.

    Parameters:
        x (array): linear space or set of values for a random variable on which the integrand is applied
        lamb (array): an array of Lagrange multipliers used to approximate the distribution
        k (integer): a constant representing the order of the moment being calculated

    Returns:
        integrand: the caclulated portion of the integrand at each x value
    '''
    neqs = len(lamb)
    xi = np.array([x**i for i in range(0, neqs)])
    return x**k * np.exp(np.dot(lamb, xi))

def residual_c_1(lamb, mu, bnds):
    '''
    Calculates the residual of the moment approximation function.
    
    Parameters:
        lamb (array): an array of Lagrange constants used to approximate the distribution
        mu (array): an array of the known moments needed to approximate the distribution function
        bnds (tuple): support bounds

    Returns:
        rhs: the integrated right hand side of the moment approximation function
    '''
    a = bnds[0]
    b = bnds[1]
    neqs = len(lamb)
    rhs = np.zeros(neqs)
    for k in range(0, neqs):
        rhs[k] = (quad(integrand_1, a, b, args=(lamb, k))[0] - mu[k])
    return rhs

def maxent_reconstruct_c0_1(mu, bnds=[0, np.inf]):
    '''
    Used to construct a continuous distribution from a limited number of known moments(μ). This function applies Maximum Entropy Theory in order to solve for the constraints found in the approximation equation that is given as an output.
    
    Parameters:
        μ: vector of size m containing the known moments of a distribution. This does NOT assume that μ0 = 1. This vector contains moments μ_k starting with μ_0, μ_1, etc...
            Ex. μ = [1,0,0]
        bnds: Support for the integration [a,b]
            ## It is important the bounds include roughly all non-zero values of the distribution that is being recreated ##
    
    Returns:
        Distribution Function: The recreated probability distribution function from the moment vector (μ) input given. requires a support to be ploted
    
    Example:
        >>> f, sol = maxent([1,0,0], [-1,1])        
    '''
    neqs = len(mu)
    lambguess = np.zeros(neqs) # initialize guesses
    lambguess[0] = -np.log(np.sqrt(2*np.pi)) # set the first initial guess - this seems to work okay
    
    
    lambsol = fsolve(residual_c_1, lambguess, args=(mu,bnds))
    print(lambsol)
    recon = lambda x: integrand_1(x, lambsol, k=0)
    return recon, lambsol

def maxent_reconstruct_c1_1(mu, bnds=[0, np.inf]):
    '''
    Used to construct a continuous distribution from a limited number of known moments(μ). This function applies Maximum Entropy Theory in order to solve for the constraints found in the approximation equation that is given as an output.
    
    Parameters:
        μ: vector of size m containing the known moments of a distribution. This does NOT assume that μ0 = 1. This vector contains moments μ_k starting with μ_0, μ_1, etc...
            Ex. μ = [1,0,0]
        bnds: Support for the integration [a,b]
            ## It is important the bounds include roughly all non-zero values of the distribution that is being recreated ##
    
    Returns:
        Distribution Function: The recreated probability distribution function from the moment vector (μ) input given. requires a support to be ploted
    
    Example:
        >>> f, sol = maxent([1,0,0], [-1,1])        
    '''
    global temp1
    
    lambguess = temp1
    lambsol = fsolve(residual_c_1, lambguess, args=(mu,bnds), full_output=True)
        
    if lambsol[2]==1 :
        print('with previous lambdas (population 1)')
        temp1=lambsol[0]
        recon = lambda x: integrand_1(x, lambsol[0], k=0)   
    
    else:
        neqs = len(mu)
        lambguess = np.zeros(neqs) # initialize guesses
        lambguess[0] = -np.log(np.sqrt(2*np.pi))
        lambsol = fsolve(residual_c_1, lambguess, args=(mu,bnds), full_output=True)    
        print('with initiall guess (population 1)')
        recon = lambda x: integrand_1(x, lambsol[0], k=0) 
        temp1=lambsol[0]

    return recon, lambsol

def integrand_2(x, lamb, k=0):
    '''
    Calculates the integrand of the \(k^\mathrm{th}\) moment.

    Parameters:
        x (array): linear space or set of values for a random variable on which the integrand is applied
        lamb (array): an array of Lagrange multipliers used to approximate the distribution
        k (integer): a constant representing the order of the moment being calculated

    Returns:
        integrand: the caclulated portion of the integrand at each x value
    '''
    neqs = len(lamb)
    xi = np.array([x**i for i in range(0, neqs)])
    return x**k * np.exp(np.dot(lamb, xi))

def residual_c_2(lamb, mu, bnds):
    '''
    Calculates the residual of the moment approximation function.
    
    Parameters:
        lamb (array): an array of Lagrange constants used to approximate the distribution
        mu (array): an array of the known moments needed to approximate the distribution function
        bnds (tuple): support bounds

    Returns:
        rhs: the integrated right hand side of the moment approximation function
    '''
    a = bnds[0]
    b = bnds[1]
    neqs = len(lamb)
    rhs = np.zeros(neqs)
    for k in range(0, neqs):
        rhs[k] = (quad(integrand_2, a, b, args=(lamb, k))[0] - mu[k])
    return rhs

def maxent_reconstruct_c0_2(mu, bnds=[0, np.inf]):
    '''
    Used to construct a continuous distribution from a limited number of known moments(μ). This function applies Maximum Entropy Theory in order to solve for the constraints found in the approximation equation that is given as an output.
    
    Parameters:
        μ: vector of size m containing the known moments of a distribution. This does NOT assume that μ0 = 1. This vector contains moments μ_k starting with μ_0, μ_1, etc...
            Ex. μ = [1,0,0]
        bnds: Support for the integration [a,b]
            ## It is important the bounds include roughly all non-zero values of the distribution that is being recreated ##
    
    Returns:
        Distribution Function: The recreated probability distribution function from the moment vector (μ) input given. requires a support to be ploted
    
    Example:
        >>> f, sol = maxent([1,0,0], [-1,1])        
    '''
    neqs = len(mu)
    lambguess = np.zeros(neqs) # initialize guesses
    lambguess[0] = -np.log(np.sqrt(2*np.pi)) # set the first initial guess - this seems to work okay
    
    
    lambsol = fsolve(residual_c_2, lambguess, args=(mu,bnds))
    print(lambsol)
    recon = lambda x: integrand_2(x, lambsol, k=0)
    return recon, lambsol

def maxent_reconstruct_c1_2(mu, bnds=[0, np.inf]):
    '''
    Used to construct a continuous distribution from a limited number of known moments(μ). This function applies Maximum Entropy Theory in order to solve for the constraints found in the approximation equation that is given as an output.
    
    Parameters:
        μ: vector of size m containing the known moments of a distribution. This does NOT assume that μ0 = 1. This vector contains moments μ_k starting with μ_0, μ_1, etc...
            Ex. μ = [1,0,0]
        bnds: Support for the integration [a,b]
            ## It is important the bounds include roughly all non-zero values of the distribution that is being recreated ##
    
    Returns:
        Distribution Function: The recreated probability distribution function from the moment vector (μ) input given. requires a support to be ploted
    
    Example:
        >>> f, sol = maxent([1,0,0], [-1,1])        
    '''
    global temp2
    
    lambguess = temp2
    lambsol = fsolve(residual_c_2, lambguess, args=(mu,bnds), full_output=True)
        
    if lambsol[2]==1 :
        print('with previous lambdas (population 2)')
        temp2=lambsol[0]
        recon = lambda x: integrand_2(x, lambsol[0], k=0)   
    
    else:
        neqs = len(mu)
        lambguess = np.zeros(neqs) # initialize guesses
        lambguess[0] = -np.log(np.sqrt(2*np.pi))
        lambsol = fsolve(residual_c_2, lambguess, args=(mu,bnds), full_output=True)    
        print('with initiall guess (population 2)')
        recon = lambda x: integrand_2(x, lambsol[0], k=0) 
        temp2=lambsol[0]

    return recon, lambsol


