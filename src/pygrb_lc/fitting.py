import numpy as np

def triple_power_law(x,A,tbr1,tbr2,alpha,beta,gamma):
    return A*np.power(x,alpha) * np.power(1+np.power(x/tbr1,w),(beta-alpha)*1/w) * np.power(1+np.power(x/tbr2,w),(gamma-beta)*1/w)

def double_power_law(x,A,tbr,alpha,beta):
    return A*np.power(x,alpha) * np.power(1+np.power(x/tbr,w),(beta - alpha)*1/w)

def power_law(x, A, alpha):
    return A*np.power(x,alpha)