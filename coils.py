import numpy as np

mur = 1
mu0 = 1

proton_freq = 128.77

def induct_solenoid(D,d,N):
    return N*N*mu0*mur*D/2.*(np.log(8.*D/d)-2)


def findL(C=1., freq = proton_freq):
    return 1/(2.*np.pi*freq)**2./C

def findC(L=1., freq = proton_freq):
    return 1/(2.*np.pi*freq)**2./L

def findnewC(C, f1 = proton_freq*.9, f_desired = proton_freq):
    return C*f1*f1/f_desired/f_desired

def findbyZ0(Z0=50., f0 = proton_freq):
    L = Z0/(2.*np.pi*f0)
    C = 1/(2.*np.pi*f0*Z0)
    return L,C



