import numpy as np

mur = 1
mu0 = 1
c = 3e8 # speed of light


proton_freq = 128.77

def findpower(voltage = None, resistance = None, current = None, power = None):
    all_vals = {'voltage': voltage,
                'resistance': resistance,
                'current': current,
                'power': power}
    if voltage is not None and resistance is not None:
        all_vals['power'] = voltage**2./resistance
    if voltage is not None and current is not None:
        all_vals['power'] = voltage*curren
    if current is not None and resistance is not None:
        all_vals['power'] = current**2.*resistance
    if voltage is not None and power is not None:
        all_vals['resistance'] = voltage**2./power
    if current is not None and resistance is not None:
        all_vals['voltage'] = current*resistance
    if voltage is not None and current is not None:
        all_vals['resistance'] = voltage/current
    if voltage is not None and resistance is not None:
        all_vals['current'] = voltage/resistance
    return all_vals

def radiation_dipole(I, length, wavelength):
    average_power = np.pi**2 / 3/c*(I*length/wavelength)**2
    return average_power

def induct_solenoid(D,d,N):
    return N*N*mu0*mur*D/2.*(np.log(8.*D/d)-2)


def findQ(bandwidth, f0 = proton_freq):
    Q = f0/bandwidth
    return Q

def findbandwidth(fhigh, flow):
    return fhigh - flow

def findreactance(C=None, L=None, R=0., f0 = proton_freq):
    XL = 2*np.pi*f0*L
    XC = 1./(2*np.pi*f0*C)
    Xtotal = XL-XC
    Z = np.sqrt(R**2+Xtotal**2)
    flow_cutoff = -R/2./L + np.sqrt(1/(L*C)+(R/2./L)**2)
    fhigh_cutoff = R/2./L + np.sqrt(1/(L*C)+(R/2./L)**2)
    Q = 2*np.pi*f0*L/R
    if XL > XC:
        return XL, XC, Z, Q, flow_cutoff, fhigh_cutoff, 'inductive'
    else:
        return XL, XC, Z, Q,f_cutoff, fhigh_cutoff, 'capacitive'

def findomega(L, C):
    omega = 1/np.sqrt(L*C)
    return omega
    
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



def microstrip(speed = None, length = None, eps_eff = 1.):
    if speed is None:
        speed = c/(2.*length *np.sqrt(eps_eff))
        return speed
    elif length is None:
        length = c/(2.*speed *np.sqrt(eps_eff))
        return length
    else:
        return 0


r_colors      = {'black':0,
                 'brown':1,
                 'red':2,
                 'orange':3,
                 'yellow':4,
                 'green':5,
                 'blue':6,
                 'violet':7,
                 'grey':8,
                 'white':9,
                 'gold':-1,
                 'silver':-2,
                 }
def resistor_value(color1 = 'black', color2 = 'black', color3 = 'black'):
    return (r_colors[color1]*10+r_colors[color2])*r_colors[color3]


def biot_savart_db(I, dl, r, J = None):
    if J is None:
        Jdl = I*dl
    else:
        Jdl = J*dl
    magnitude = np.sqrt(r[0]**2. + r[1]**2. + r[2]**2.)
    factor = mu/4./np.pi/magnitude
    s1 = Jdl[1]*r[2] - Jdl[2]*r[1]
    s2 = Jdl[2]*r[0] - Jdl[0]*r[2]
    s3 = Jdl[0]*r[1] - Jdl[1]*r[2]
    s1 = s1/factor
    s2 = s2/factor
    s3 = s3/factor
    return s1, s2, s3


def VSWR(S11=0):
    VSWR = (1+np.abs(S11))/(1-np.abs(S11))
    return VSWR

def amplifier_stability(S11, S12, S21, S22):
    Delta = S11*S22-S12*S21
    center = np.conj(S11-Delta*np.conj(S22))/(np.abs(S11)**2-np.abs(Delta)**2)
    radius = np.abs(S12*S21/(np.abs(S11)**2-np.abs(Delta)**2))
    rollet = (1-np.abs(S11)**2-np.abs(S22)**2-np.abs(Delta)**2)/(2.*np.abs(S12*S21))
    is_stable = np.abs(K) > 1 and np.abs(Delta) < 1
    return center, radius, Delta, rollet, is_stable



def z_to_s(Zload, Z0=50.):
    S11 = (Zload - Z0)/(Zload+Z0)
    return S11

def gamma_to_z(gamma = 0, Z0 = 50.):
    Z = Z0 * (1+gamma)/(1-gamma)
    return Z


