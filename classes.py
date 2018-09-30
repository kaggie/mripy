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

        
        
        

class multinuclear():
    def __init(self):
        gyros = """Isotope      Symbol  Name    Spin    Natural
        Abund. %        Receptivity
        (rel. to 13C)   Magnetic
        Moment  Gamma
        (x 10^7
        rad/Ts)         Quadrup.
        Moment
        Q/fm^2  Frequency       Reference
        191     Ir      Iridium 3/2     37.30000        0.06412         0.19460         0.48120         81.60000        6.872
        197     Au      Gold    3/2     100.00000       0.16294         0.19127         0.47306         54.70000        6.916
        235     U       Uranium 7/2     0.72000 ---     -0.43000        -0.52000        493.60000       7.366
        193     Ir      Iridium 3/2     62.70000        0.13765         0.21130         0.52270         75.10000        7.484
        187     Os      Osmium  1/2     1.96000 0.00143         0.11198         0.61929         ---     9.129   OsO4
        179     Hf      Hafnium 9/2     13.62000        0.43824         -0.70850        -0.68210        379.30000       10.068
        41      K       Potassium       3/2     6.73020 0.03341         0.27740         0.68607         7.11000         10.245  KCl
        167     Er      Erbium  7/2     22.93000        ---     -0.63935        -0.77157        356.50000       11.520
        155     Gd      Gadolinium      3/2     14.80000        ---     -0.33208        -0.82132        127.00000       12.280
        103     Rh      Rhodium 1/2     100.00000       0.18600         -0.15310        -0.84680        ---     12.746  Rh(acac)3 p
        57      Fe      Iron    1/2     2.11900 0.00425         0.15696         0.86806         ---     12.951  Fe(CO)5
        145     Nd      Neodymium       7/2     8.30000 ---     -0.74400        -0.89800        -33.00000       13.440
        161     Dy      Dysprosium      5/2     18.91000        ---     -0.56830        -0.92010        250.70000       13.760
        149     Sm      Samarium        7/2     13.82000        ---     -0.76160        -0.91920        7.40000         13.760
        73      Ge      Germanium       9/2     7.73000 0.64118         -0.97229        -0.93603        -19.60000       13.953  (CH3)4Ge
        83      Kr      Krypton 9/2     11.49000        1.28235         -1.07311        -1.03310        25.90000        15.390  Kr
        177     Hf      Hafnium 7/2     18.60000        1.53529         0.89970         1.08600         336.50000       16.028
        157     Gd      Gadolinium      3/2     15.65000        ---     -0.43540        -1.07690        135.00000       16.120
        107     Ag      Silver  1/2     51.83900        0.20500         -0.19690        -1.08892        ---     16.191  AgNO3
        183     W       Tungsten        1/2     14.31000        0.06310         0.20401         1.12824         ---     16.666  Na2WO4
        147     Sm      Samarium        7/2     14.99000        ---     -0.92390        -1.11500        -25.90000       16.680
        87      Sr      Strontium       9/2     7.00000 1.11765         -1.20902        -1.16394        33.50000        17.335  SrCl2
        105     Pd      Palladium       5/2     22.33000        1.48824         -0.76000        -1.23000        66.00000        18.304  K2PdCl6
        99      Ru      Ruthenium       5/2     12.76000        0.84706         -0.75880        -1.22900        7.90000         18.421  K4[Ru(CN)6]
        109     Ag      Silver  1/2     48.16100        0.29000         -0.22636        -1.25186        ---     18.614  AgNO3
        39      K       Potassium       3/2     93.25810        2.80000         0.50543         1.25006         5.85000         18.665  KCl
        163     Dy      Dysprosium      5/2     24.90000        ---     0.79580         1.28900         264.80000       19.280
        173     Yb      Ytterbium       5/2     16.13000        ---     -0.80446        -1.30250        280.00000       19.284
        89      Y       Yttrium 1/2     100.00000       0.70000         -0.23801        -1.31628        ---     19.601  Y(NO3)3
        101     Ru      Ruthenium       5/2     17.06000        1.59412         -0.85050        -1.37700        45.70000        20.645  K4[Ru(CN)6]
        143     Nd      Neodymium       7/2     12.20000        ---     -1.20800        -1.45700        -63.00000       21.800
        47      Ti      Titanium        5/2     7.44000 0.91765         -0.93294        -1.51050        30.20000        22.550  TiCl4
        49      Ti      Titanium        7/2     5.41000 1.20588         -1.25201        -1.51095        24.70000        22.556  TiCl4
        53      Cr      Chromium        3/2     9.50100 0.50765         -0.61263        -1.51520        -15.00000       22.610  K2CrO4
        40      K       Potassium       4       0.01170 0.00360         -1.45132        -1.55429        -7.30000        23.208  KCl
        25      Mg      Magnesium       5/2     10.00000        1.57647         -1.01220        -1.63887        19.94000        24.487  MgCl2
        67      Zn      Zinc    5/2     4.10000 0.69412         1.03556         1.67669         15.00000        25.027  Zn(NO3)2
        95      Mo      Molybdenium     5/2     15.92000        3.06471         -1.08200        -1.75100        -2.20000        26.068  Na2MoO4
        201     Hg      Mercury 3/2     13.18000        1.15882         -0.72325        -1.78877        38.60000        26.446  (CH3)2HgD
        97      Mo      Molybdenium     5/2     9.55000 1.95882         -1.10500        -1.78800        25.50000        26.615  Na2MoO4
        43      Ca      Calcium 7/2     0.13500 0.05106         -1.49407        -1.80307        -4.08000        26.920  CaCl2
        14      N       Nitrogen        1       99.63200        5.88235         0.57100         1.93378         2.04400         28.905  CH3NO2
        33      S       Sulfur  3/2     0.76000 0.10118         0.83117         2.05568         -6.78000        30.704  (NH4)2SO4
        189     Os      Osmium  3/2     16.15000        2.32353         0.85197         2.10713         85.60000        31.062  OsO4
        21      Ne      Neon    3/2     0.27000 0.03912         -0.85438        -2.11308        10.15500        31.577  Ne
        176     Lu      Lutetium        7       2.59000 ---     3.38800         2.16840         497.00000       32.524
        37      Cl      Chlorine        3/2     24.22000        3.87647         0.88320         2.18437         -6.43500        32.623  NaCl
        131     Xe      Xenon   3/2     21.18000        3.50588         0.89319         2.20908         -11.40000       32.976  XeOF4
        169     Tm      Thulium 1/2     100.00000       ---     -0.40110        -2.21800        ---     33.160
        61      Ni      Nickel  3/2     1.13990 0.24059         -0.96827        -2.39480        16.20000        35.744  Ni(CO)4
        91      Zr      Zirconium       5/2     11.22000        6.29412         -1.54246        -2.49743        -17.60000       37.185  Zr(C5H5)2Cl2        """

        def sep(text):
            lines=gyros.split('\n')
            dicts = lines[0].split('\t')
            alldata = []
            for line in lines:
                entries = line.split('\t')
                mydict = {}
                print dicts, len(dicts), len(entries)
                if len(entries) > len(dicts):
                    for xi in xrange(len(dicts)):
                        mydict[dicts[xi]] = entries[xi]
                        print dicts[xi], entries[xi]
                alldata.append(mydict)
            return alldata

        alldata = sep(gyros)
        
