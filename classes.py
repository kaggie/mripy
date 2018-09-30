  class datastore():
    def __init__(self,parent=None, imdata = None, name = '', datatype = None, scanparams = None, plotdata = None, y = None, x = None,
                 TEs = None, TSLs = None, TRs = None, T1s = None, T2s = None,
                 B0 = None, FlipAngle = None, M0 = None, position = (None,None,None), rotation = None,
                 FOV = (None, None, None), voxelsize = (None, None, None), nucleus = None, zoom = (None, None, None), TxVoltage = None, dicominfo = None,
                 itype = None, metadata = None, preferredIndex = None, preferredplotSettings = {}):
        self.parent = parent
        self.dims = {} # 'nx','ny','nz','te','ncest','ntsl','ntr'
        self.imdata = np.array(imdata)
        self.datatype = datatype
        self.name = name
        self.plotdata = plotdata #{'x':, 'y':, 'rho':, 'theta':, }
        self.scanparams = scanparams
        self.metadata = metadata
        self.y = y
        self.x = x
        self.TEs = TEs
        self.TSLs = TSLs
        self.TRs = TRs
        self.T1s = T1s
        self.T2s = T2s
        self.B0 = B0
        self.FlipAngle = FlipAngle
        self.M0 = M0
        self.Mz = None
        self.Mx = None
        self.My = None
        self.gradB0 = None
        self.T1 = None
        self.T2 = None
        self.T2star = None
        self.T1rho = None
        self.preferredIndex = preferredIndex
        self.preferredplotSettings = preferredplotSettings
        self.phase = None
        self.unwrappedphase = None
        self.elasticity = None
        self.DWI = None
        self.DTI = None
        self.position = position
        self.rotation = rotation
        self.FOV = FOV
        self.voxelsize = voxelsize
        self.nucleus = nucleus
        self.zoom = zoom
        self.TxVoltage = TxVoltage
        self.dicominfo = dicominfo
        self.gradientx = None
        self.gradientz = None
        self.gradienty = None
        self.conductivity = None
        self.Temp = None
        self.B1 = None
        self.EMfield = None
        self.pulsesequence = None
        self.coils = None
        self.permittivity = None
        self.permeability = None
        self.voxelsensitivity = None
        self.biotsavartfield = None
    def return_name(self, increment=False):
        if self.name == None:
            if len(np.shape(self.imdata))>0:

                self.parent.datastorecount = self.parent.datastorecount+1
                self.name = str(self.parent.datastorecount).zfill(2)+': ' + str(np.shape(self.imdata))
            else:
                pass
        if increment == True:
            self.parent.datastorecount = self.parent.datastorecount+1
            self.name = str(self.parent.datastorecount)+ self.name[3:]
        return self.name
    def copy(self, datastoretocopy):
        #self.parent = datastoretocopy.parent
        self.imdata = np.array(datastoretocopy.imdata)
        self.datatype = datastoretocopy.datatype
        self.name = datastoretocopy.name
        self.plotdata = datastoretocopy.plotdata
        self.y = datastoretocopy.y
        self.x = datastoretocopy.x
        self.TEs = datastoretocopy.TEs
        self.TSLs = datastoretocopy.TSLs
        self.TRs = datastoretocopy.TRs
        self.T1s = datastoretocopy.T1s
        self.T2s = datastoretocopy.T2s
        self.B0 = datastoretocopy.B0
        self.FlipAngle = datastoretocopy.FlipAngle
        self.M0 = datastoretocopy.M0
        self.position = datastoretocopy.position
        self.rotation = datastoretocopy.rotation
        self.FOV = datastoretocopy.FOV
        self.voxelsize = datastoretocopy.voxelsize
        self.nucleus = datastoretocopy.nucleus
        self.zoom = datastoretocopy.zoom
        self.TxVoltage = datastoretocopy.TxVoltage
        self.dicominfo = datastoretocopy.dicominfo
        self.preferredIndex = datastoretocopy.preferredIndex
        self.preferredplotSettings = datastoretocopy.preferredplotSettings
