## Some very basic MRI functions.
## A random assortment of tools

import numpy as np

class gradient_class():
    def __init__(self,k=np.array(10)):
        self.k = k
    def calc_gradient(self):
        self.gradient = np.gradient(self.k)
        self.slew = np.gradient(self.gradient)
    def calc_k(self):
        self.gradient = np.cumsum(self.slew)
        self.k = np.cumsum(self.gradient)
    def slewlimit(self, slewmax=.1):
        slew = self.slew
        slewnew = np.array(slew)
        slewnew[slew>slewmax] = slewmax
        slewnew[slew<-slewmax] = -slewmax
        self.slew = slewnew
        return slewnew
    def gradlimit(self, gradmax=.1):
        grad = self.gradient
        grad0 = np.abs(grad)<np.abs(gradmax)
        gradnew = np.array(grad)
        gradnew[grad>gradmax] = gradmax
        gradnew[grad<-gradmax] = -gradmax
        self.gradient = gradnew
        return gradnew
    def ramp(vstart=0, vend = 1, vmax = .2,vdotmax=.2):
        # v could be gradient, or kspace.
        # vdot means the derivative of v, like slew rate or gradient
        vdiff = vend - vstart
        nvt = vdiff/vdotmax
        vdots = np.arange(nvt)*vdotmax        
        v = np.cumsum(vdots)    
        v[v>vmax] = vmax
        vsum = np.cumsum(v)
        vdots = np.gradient(v)
        return v, vdots, vsum # grads, k, slew

            
def ernst(t1=0,tr=0,alpha=0,alpharad = 0):
    varA = -1
    if alpha==0 and alpharad==0: #solve for alpha
        varA = np.arccos(np.exp(-tr/t1))*180/np.pi   #returns value in degrees
    elif tr==0: #solver for tr
        if alpharad==0:
            alpharad = alpha*np.pi/180
        varA = -t1*np.log(np.cos(alpharad))
    elif t1==0: #solve for t1
        if alpharad ==0:
            alpharad = alpha*np.pi/180
        varA = -tr/np.log(np.cos(alpharad))
    else:
        print('not enough defined values to calculate!')
    return varA
            
            
def zerofill(im,padx,pady):
    "zero fills both sides of an image"
    im = np.array(im)
    try:
        newim = np.zeros((im.shape[0]+2*padx,im.shape[1]+2*pady),dtype=complex)
        newim[padx:(im.shape[0]+padx),pady:(im.shape[1]+pady)] = im
    except:
        newim = np.zeros((im.shape[0]+2*padx,im.shape[1]+2*pady,im.shape[2]),dtype=complex)
        newim[padx:(im.shape[0]+padx),pady:(im.shape[1]+pady),:] = im
    return newim

#import sp.ndimage.interpolation.zoom as zoom



def zerointerp(im,padx=0,pady=0,zoom=0):
    "returns a kspace zero-filled interpolated image"
    im = np.array(im)
    newim = im
    if padx>0 or pady>0 or zoom >0:
        if zoom!=0:
            padx = im.shape[0]*zoom/2
            pady = im.shape[1]*zoom/2
            im = im*(zoom+1)**2
        newim = np.fft.ifft2(zerofill(np.fft.fftshift(np.fft.fft2(im,axes=(0,1)),axes=(0,1)),padx,pady),axes=(0,1))
    elif padx<0 or pady<0 or zoom <0:
        if zoom!=0:
            padx = np.int(im.shape[0]/zoom/2)
            pady = np.int(im.shape[1]/zoom/2  )
        newim = np.fft.ifft2(np.fft.fftshift(np.fft.fft2(im,axes=(0,1)),axes=(0,1))[-padx:padx,-pady:pady],axes=(0,1))
    return newim            
            
            
from numpy.fft import fftshift

def im2kspace(im):
    return fftshift(np.fft.fft2(im,axes=(0,1)),axes=(0,1))

def kspace2im(ksp):
    return np.fft.ifft2(ksp,axes=(0,1))

def kspace3d2im(ksp):
    return np.fft.ifftn(ksp,axes=(0,1,2))


class shapes():
    "returns a shape that can then be used as a filter"
    def __init__ (self,Nx,Ny=1,Nz=1,offsetx=.12345, offsety=.12345, offsetz=.12345, scalex=1, scaley=1,scalez=1):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        if offsetx ==.12345:  #just an unlikely number that no one would use
            offsetx = Nx/2
        if offsety ==.12345:
            offsety = Nx/2
        if offsetz ==.12345 and Nz>1:
            offsetz = Nx/2
        elif Nz ==1:
            offsetz = 0
        #if Nz>1:
        vectorz = np.arange(Nz)
        #else:
        #    vectorz = np.arange(1)+1
        self.meshx,self.meshy,self.meshz = np.meshgrid(np.arange(Nx)-offsetx,np.arange(Ny)-offsety,vectorz-offsetz)
        self.meshx = self.meshx/scalex;self.meshy = self.meshy/scaley;self.meshz = self.meshz/scalez;
        self.meshr = np.sqrt(self.meshx**2+self.meshy**2+self.meshz**2)
    def circle(self,radius=1):
        self.out = np.zeros((self.Nx,self.Ny,self.Nz))
        self.out[self.meshr<radius] = 1
        return np.squeeze(self.out)
    def square(self,lengthx=1, lengthy=1):
        self.out = np.zeros((self.Nx,self.Ny, self.Nz))
        self.out[self.meshx<lengthx] = 1
        self.out[self.meshy<lengthy] = 1
        return np.squeeze(self.out)


def dualanglemap(imalpha, im2alpha):
    #creates a flip angle map from two images
    im1 = np.abs(imalpha)
    im2 = np.abs(im2alpha)
    alpha = np.abs(np.arccos(im2/im1/2)*180/np.pi)
    alpha[np.isnan(alpha)] = 0
    return alpha
    
def strip_filepath(filename):
    path = '/'.join(filename.split('/')[:-1])+'/'
    return path
    
    
def combined_phasemap(imSpace,TE):
    try:
        return combined_phasemap4(imSpace,TE)
    except:
        return combined_phasemap5(imSpace,TE)

def combined_phasemap4(imSpace, TE, unwrapstyle = 'unwrap'):
    "complex imSpace[ y, x, te, coil ]; TE[0], TE[1]"
    "according to Robinson, MRM 65:1638(2011)"
    imSpace = np.transpose(imSpace,[0,1,3,2])
    phase = np.angle(imSpace)
    tediff = TE[0]-TE[1]
    if unwrapstyle == 'unwrap':
        theta_rx = np.array((TE[0]*unwrap3D(phase[:,:,1,:])-TE[1]*unwrap3D(phase[:,:,0,:]))/tediff)
    elif unwrapstyle =='punwrap':
        theta_rx = np.array((TE[0]*punwrap(imSpace[:,:,1,:])-TE[1]*punwrap(imSpace[:,:,0,:]))/tediff)
 #make sure this unwraps each coil individually
    else:
        theta_rx = np.array((TE[0]*unwrap3D(phase[:,:,1,:])-TE[1]*unwrap3D(phase[:,:,0,:]))/tediff)
    theta_rx.shape
    im = weightbynoise(imSpace)
    meh = np.transpose(im,[2,0,1,3])
    theta_wm = np.sum(meh*np.exp(-1j*theta_rx),axis=3)
    meh2 = np.transpose(theta_wm,[1,2,0])
    return np.angle(meh2)
    

def weightmap(imweight, immap, dimension = 2):
    "returns the summation over a dimension, with elements from that dimension given weights"
    "default of along the 3rd dimension (dimension=2)"
    np.ma.masked_array(immap)
    imweight = np.abs(1/imweight)**2
    returnthis = np.sqrt(np.sum(imweight*immap,dimension)/np.sum(imweight,dimension))
    returnthis[np.isnan(returnthis)] = 0
    return returnthis


def combine_sos(imSpace, coilpos = 3,useabs=True):
    return sos_image(imSpace, coilpos, useabs)

def sos_image(imSpace, coilpos = 3,useabs=True):
    "default imSpace[ y, x, te, coil ]"
    #if len(imSpace.shape)==4:
    if useabs:
        im = np.abs(weightbynoise(np.abs(imSpace)))
    else:
        im = imSpace
    return np.sqrt(np.mean(im**2,coilpos))

def combined_image(imspace, TE = [2.69, 5.86],coilpos=2):
    phaseimage = combined_phasemap(imspace,TE)
    sosimage = sos_image(imspace,coilpos)
    return sosimage*np.exp(1j*phaseimage)
    
    
    
def combined_b0map(imSpace,TE=[-100,100],unwrapstyle = 'unwrap'):
    "complex imSpace[ y, x, te, coil ]; TE[0], TE[1]"
    "according to Josh Kaggie"
    im = weightbynoise(imSpace)
    tediff = TE[1]-TE[0]
    gamma = 42.596
    factor = 2*np.pi*tediff*gamma
    if tediff==200:
        factor = 1
    b0_coils = np.angle(im[:,:,0,:]/im[:,:,1,:])
    meh = np.transpose(im,[2,0,1,3])
    sum_b0_sig = np.angle(np.sum(np.exp(1j*b0_coils)*np.abs(meh)**2, axis=3))
    b0 = sum_b0_sig[0,:,:]
    return b0/factor

def combined_t2map(imSpace,t2map , threshhold = 7):
    "complex imSpace[ y, x, te, coil ]; TE[0], TE[1]"
    "according to Josh Kaggie"
    im = np.abs(weightbynoise(imSpace)[:,:,0,:]) #remove the matrix elements....
    t2map[np.nonzero(t2map>1000)]=0
    im[np.nonzero(im<threshhold)]=1e-20
    t2map[np.nonzero(im<3)]=0.0001
    r2map = 1/t2map
    sum_im2_r2 = np.sum(r2map*im**1, axis=2)
    sum_im2 = np.sum(im**1, axis=2)
    r2wmap = sum_im2_r2/sum_im2
    return 1/r2wmap


def gradient(b0):
    grad = np.gradient(b0)
    return np.sqrt(grad[0]**2+grad[1]**2)

def im_filter(im, stddev=0.5):
    import scipy.ndimage as snd
    return snd.gaussian_filter(im,stddev)




def r(im):
    "rotation"
    if len(im.shape)==2:
        b = np.transpose(im,[1,0])
    elif len(im.shape)==3:
        b = np.transpose(im,[1,0,2])
    elif len(im.shape)==4:
        b = np.transpose(im,[1,0,2,3])
    elif len(im.shape)==5:
        b = np.transpose(im,[1,0,2,3,4])
    else:
        b = im
    return b
    
    
    
def noisecorrxy(x,y):
    meanx = np.mean(x)
    meany = np.mean(y)
    stdx = np.std(x)
    stdy = np.std(y)
    noisecorr = np.mean((x-meanx)*np.conj(y-meany))/stdx/stdy
    return noisecorr

def noisecorrmat(data):
    someshape = np.shape(data)[-1]
    noisecorr = np.zeros((someshape,someshape))
    for xi in xrange(someshape):
        datax = (data[...,xi])
        for yi in xrange(someshape):
            datay = (data[...,yi])
            noisecorr[xi,yi] = noisecorrxy(datax,datay)
    return noisecorr



def shiftimage(im,shifty=0,shiftx = 0, shiftz=0 ):
    newimage = np.array(im)
    if np.abs(shifty)>0:
        newimage[:shifty] = im[-shifty:]
        newimage[shifty:] = im[:-shifty]
    im = np.array(newimage)
    if np.abs(shiftx)>0:
        newimage[:,:shiftx] = im[:,-shiftx:]
        newimage[:,shiftx:] = im[:,:-shiftx]
    im = np.array(newimage)
    if np.abs(shiftz)>0:
        newimage[:,:,:shiftz] = im[:,:,-shiftz:]
        newimage[:,:,shiftz:] = im[:,:,:-shiftz]
    return newimage
    
    
def dixon2dicom(TE1=4.7, TE2=5.75, TE3=6.8, filename1='', filename2='', filename3=''):
    filename1,path1 =  _checkloadfilename(filename1,title='Open TE1')
    fat,water,b0 = dixonfiles(TE1=TE1, TE2=TE2, TE3=TE3, filename1=filename1,\
                            filename2=filename2, filename3=filename3)
    fat = fat[64:192]
    water = water[64:192]
    fatdicomclass = vol2dicom(fliplr(r(fat)),justdoit=False,reducebyroi=False,studytype='fat')
    fatdicomclass.defpath = path1
    dicomfilename =  _checkloadfilename(initialdir=fatdicomclass.defpath, title='Open Example Dicom File',filetypes=[('Dicom', '.IMA'),('all','')])
    fatdicomclass.getdicom(dicomfilename[0])
    savefatas = _checksavefilename(initialdir=fatdicomclass.defpath,title='Save Fat Dicoms as')
    fatdicomclass.dicominfo.ProtocolName = '3 Point Dixon - Fat'
    fatdicomclass.writeslices(savefatas[0])
    waterdicomclass = vol2dicom(fliplr(r(water)),justdoit=False,reducebyroi=False,studytype='fat')
    waterdicomclass.defpath = fatdicomclass.defpath
    waterdicomclass.dicominfo = fatdicomclass.dicominfo
    waterdicomclass.dicominfo.ProtocolName = '3 Point Dixon - Water'
    waterdicomclass.dicominfo.SeriesDescription = 'Water'
    waterdicomclass.dicominfo.SeriesNumber = 79
    savewateras = _checksavefilename(initialdir=waterdicomclass.defpath,title='Save Water Dicoms as')
    waterdicomclass.writeslices(savewateras[0])






def dixon(acq1,acq2,acq3,teinc1,teinc2):
  def dixon4d(acq1,acq2,acq3,teinc1,teinc2):
      fat = []
      water = []
      b0 = []
      for xi in xrange(np.shape(acq1)[3]):
          fata,watera,b0a = dixon2d3d(acq1[:,:,:,xi], acq2[:,:,:,xi], acq3[:,:,:,xi], teinc1, teinc2)
          fat.append(fata)
          water.append(watera)
          b0.append(b0a)
      fat = np.array(fat)
      water = np.array(water)
      fat = np.sqrt(np.sum(np.abs(fat)**2,axis=0))
      water = np.sqrt(np.sum(np.abs(water)**2,axis=0))
      return fat,water,b0
  def dixon2d3d(acq1, acq2, acq3, teinc1, teinc2, freqshift = 444):
      b0 = unwrap(np.angle(acq3 / acq1)) * 1000 / teinc2 / (2*np.pi);
      acq1 = np.array(np.array(acq1))
      acq2 = np.array(np.array(acq2))
      acq3 = np.array(np.array(acq3))
      #b0 = np.angle(acq3 / acq1) * 1000 / teinc2 / (2*np.pi);  #doesn't use gammabar, as it cancels out in D
      fat = 0;
      water = 0;
      B = np.exp(1j*freqshift*teinc1*1e-3*2*np.pi);
      D = np.exp(-1j*b0*teinc1*1e-3*2*np.pi);
      water = (acq1 - acq2*B*D)/(1-B);
      fat = acq1 - water;
      return fat,water,b0
  if len(np.shape(acq1))==4:
      fat,water,b0 = dixon4d(acq1, acq2, acq3, teinc1, teinc2)
  else:
      fat,water,b0 = dixon2d3d(acq1, acq2, acq3, teinc1, teinc2)
  return fat,water,b0
  
  


def weightbynoise(imSpace):
    "returns imSpace, by SNR, calculated by the corners of the image"
    imSpace = np.array(imSpace)
    stdd=np.zeros([9,1])
    try:
        for xi in xrange(imSpace.shape[3]):
            stdd[0] = np.std(imSpace[:,:,0,xi])
            stdd[1] = np.std(imSpace[0:20,0:20,0,xi])
            stdd[2 ]= np.std(imSpace[0:20,-20:,0,xi])
            stdd[3] = np.std(imSpace[-20:,0:20,0,xi])
            stdd[4] = np.std(imSpace[-20:,-20:,0,xi])
            stdd[5] = np.std(imSpace[-10:,:,0,xi])
            stdd[6] = np.std(imSpace[:10,:,0,xi])
            stdd[7] = np.std(imSpace[:,:10,0,xi])
            stdd[8] = np.std(imSpace[:,-10:,0,xi])
            stddev = np.min(stdd)
            if stddev==0:
                stddev = 1
            imSpace[:,:,:,xi] = imSpace[:,:,:,xi]/stddev
    except: #no coils
        if 1: #for xi in xrange(imSpace.shape[3]):
            stdd[0] = np.std(imSpace[:,:,0])
            stdd[1] = np.std(imSpace[0:20,0:20,0])
            stdd[2 ]= np.std(imSpace[0:20,-20:,0])
            stdd[3] = np.std(imSpace[-20:,0:20,0])
            stdd[4] = np.std(imSpace[-20:,-20:,0])
            stdd[5] = np.std(imSpace[-10:,:,0])
            stdd[6] = np.std(imSpace[:10,:,0])
            stdd[7] = np.std(imSpace[:,:10,0])
            stdd[8] = np.std(imSpace[:,-10:,0])
            stddev = np.min(stdd)
            if stddev==0:
                stddev = 1
            imSpace[:,:,:] = imSpace[:,:,:]/stddev
    return imSpace
    

def t2eqn(x,a,t2):
    return (1/t2-a*x)**(-1)

def fit_data(points, times, which='fexp'):
    from scipy.optimize import curve_fit
    points = np.asarray(np.abs(points))
    times = np.asarray(times)
    def flinear(x,a,b):
        return a+x*b
    def fexp(x,a,b):
        return a*np.exp(-x/b)
    def fexpnoise(x,a,b,c):
        return a*np.exp(-x/b)+c
    def fdoubleexp(x,a,b,c,d):
        return a*np.exp(-x/b)+c*np.exp(-x/d)
    def inverseeqn(x,a):
        return a/x
    def inversion(x,a,b):
        return a*(1-np.exp(-x/b))
    popt, pcov  = curve_fit(eval(which), times, points)
    return popt

def fit_loop_nocoils(imspace, TEs, which='fexp'):
    "imspace[x, y, TE, coil"
    yi = 0; ci = 0; ai = 0;xi=0;
    imspace = np.abs(imspace)
    li = np.array(imspace).shape
    llength = (li[0],li[1])
    Aarray = np.zeros((llength))
    Barray = np.zeros((llength))
    print Aarray.shape
    print Barray.shape
    for ci in xrange(0, (imspace.shape)[3]):
        print( li[3]-ci, li[1]-yi, li[0]-xi, ai)
        for yi in range(0, (imspace.shape)[1]):
            print( li[3]-ci, li[1]-yi, li[0]-xi, ai,)
            for xi in xrange(0, (imspace.shape)[0]):
                #print imspace.shape
                if 1:
                #for ai in xrange(0, 1): #length(imspace)[3]):

                    points = imspace[xi,yi,:,ci]
                    try:
                        popt = fit_data(points,TEs, which)
                    except KeyboardInterrupt:
                        sys.exit()
                        #return Aarray, Barray
                    except:
                        popt = -0.0001,-0.0001
                    Aarray[xi, yi, ci] = popt[0]
                    Barray[xi, yi, ci] = popt[1]
    return Aarray, Barray



def fit_linear_t1(im_array, flip_angles,TR,expon = True,magthreshlow = 2, t1threshlow=0, t1threshhigh= 10000):
    # y = a+bx
    im_array = np.abs(im_array).astype(np.float)
    Si_sin_alpha = im_array/np.sin(flip_angles).astype(np.float)
    Si_tan_alpha = im_array/np.tan(flip_angles).astype(np.float)
    #Si_sin_alpha, Si_tan_alpha
    y = Si_sin_alpha/100000.
    x = Si_tan_alpha/100000.

    sxy = np.sum(x*y, axis=-1).astype(np.float)
    sxx = np.sum(x**2, axis=-1).astype(np.float)
    sx = np.sum(x, axis=-1).astype(np.float)
    sy = np.sum(y, axis=-1).astype(np.float)
    sx2 = sx**2
    n = np.shape(flip_angles)[-1]
    b = (n*sxy-sx*sy)/(n*sxx-sx2)
    a = ((sy-b*sx)/n)
    t1 = -TR/np.log(b)/100000.*1000
    t1[t1<t1threshlow] = 0
    t1[t1>t1threshhigh] = 0
    return t1 #{'y':y,'x':x,'sxy':sxy,'sxx':sxx,'sx':sx,'sy':sy,'sx2':sx2,'n':n,'b':b,'a':a,'t1':t1}


#findminofarray(x,y,polydim,sizeofsides)
#finds the poly interpolated minimum of x and y, to the order
#of polydim, keeping sizesofsides on both sides of the initial minimum in the matrix
def findminofarray(x,y,polydim=3,sizeofsides=2,verbose=False):
    themin = np.argmin(y)
    mina = themin-sizeofsides
    maxa = themin+sizeofsides
    if mina<0:
        mina = 0
    if maxa>len(x):
        maxa = len(x)
    x = np.array(x[mina:maxa])
    y = np.array(y[mina:maxa])
    if not verbose:
        import warnings
        warnings.simplefilter('ignore', np.RankWarning)
    z = np.polyfit(x,y,polydim)
    a = 3*z[0]; b = 2*z[1]; c = z[2]
    xroot1 = (-b+np.sqrt(b**2-4*a*c))/2/a
    xroot2 = (-b-np.sqrt(b**2-4*a*c))/2/a
    xroot = xroot1
    if np.abs(xroot1-themin) > np.abs(xroot2-themin):
        xroot = xroot2
    return xroot
    
 
 
def averagemats(filenames = [], thekey='immat',makeabs = False):
    if len(filenames) == 0:
        filenames = getfiles(initialdir = defpath,useQT = True)
    collection = []
    for thisfile in filenames:
        collection.append(sio.loadmat(thisfile)[thekey])
    collection = np.array(collection)
    if makeabs:
        collection = np.abs(collection)
    return np.mean(collection,0)
    
    
    
class imspacedata():
    def __init__(self,imspace=[0]):
        self.imspace = imspace
    

    
    
def basicregrid(data, ktrajs,divideby=256 , resolution=[64.,64.,64.], method='nearest'):
    #also method='linear', 'cubic'; method='nearest',
    if data.shape == ktrajs[0].shape:
        from scipy.interpolate import griddata
        data = np.array(data).flatten()
        ktrajx = np.array(ktrajs[0]).flatten()/divideby
        ktrajy = np.array(ktrajs[1]).flatten()/divideby
        ktrajz = np.array(ktrajs[2]).flatten()/divideby
        points = np.transpose(np.array([ktrajx, ktrajy, ktrajz]))
        stepx = 1/resolution[0]
        #stepx = .05
        gridx = np.arange(ktrajx.min(),ktrajx.max(),stepx)
        stepy = 1/resolution[1]
        #stepy = .05
        gridy = np.arange(ktrajy.min(),ktrajy.max(),stepy)
        stepz = 1/resolution[2]
        #stepz = .05
        gridz = np.arange(ktrajz.min(),ktrajz.max(),stepz)
        meshx, meshy, meshz = np.meshgrid(gridx, gridy, gridz)

        regridded = griddata(points,data,(meshx,meshy,meshz),method=method)
    else:
        regridded = -1
        print( "shapes are different")
    return regridded



    
    
    
    
    





            
            
