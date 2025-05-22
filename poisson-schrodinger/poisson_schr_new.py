# Poisson-Schroedinger solver with OO interface
# see poisson_schr3.py for earlier, less flexible non-OO interface.

import scipy as sp
import scipy.constants as cn
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg




class poiss_schr:
    

    def __init__(self):
        self.layers = []
    
    def addlayer(self, d = 0., Ec = 0., n = 0., epsr = 0., x = None):
        if x != None :
            Ec = .79 * x
        self.layers.append((d, Ec, n, epsr))
  
    
    def setup(self, qr, T = 20, Vbias = (.76, .1), Edx = -.12, meff = .067, h = 1, nev=5):
    # def setup(self, qr, T = 20, Vbias = (.76, .1), Edx = -.12, epsr = 13, meff = .067, h = 1, nev=5):
        
        self.r = np.array(self.layers, dtype=[('d', float), ('Ec', float), ('n', float), ('epsr', float)])
        r = self.r        
        self.qr = qr        
        self.h = h
        self.meff = meff
        # self.epsr = epsr
        self.epsr = r[:]['epsr'][qr]
        self.Edx = Edx
        # (x-.22)*0.65 according to Fig 7 of Mooney review        
        
        self.Vbias = Vbias
        
        self.nev = nev        
        
        self.w = np.sum(r[:]['d'])
        
        self.z = np.arange(0, self.w, h)
        self.npts = len(self.z)
        self.Vband = np.zeros(self.npts)
        self.nd = np.zeros(self.npts)
        
        self.epsr_vec = []
        for ii in range(len(r[:]['d'])):
            self.epsr_vec.append(np.multiply(np.ones(int(r[:]['d'][ii]/h)), r[:]['epsr'][ii]))
        
        self.epsr_vec = np.concatenate(self.epsr_vec) 
        
        self.zinter = np.cumsum(r[:]['d'])
        for i in range(0,len(r)):
            mask = (self.z > self.zinter[i]-r[i]['d']) & (self.z <= self.zinter[i])
            self.Vband[mask] = r[i]['Ec']
            self.nd[mask] = r[i]['n']*1e-21 # convert to charges/nm^
        
        
        self.qind = (np.argmin(np.abs(self.z-self.zinter[qr-1])), np.argmin(np.abs(self.z-self.zinter[qr])))
        self.nqu = self.qind[1] - self.qind[0]
                
        # Laplace matrix for quantum problem
        # work in nm and eV
        E0 = .5*(1e9*cn.hbar/h)**2/(cn.e * cn.m_e*meff)
        self.Q0 = E0 * sp.sparse.diags([2*np.ones(self.nqu), -np.ones(self.nqu-1), -np.ones(self.nqu-1)], [0, 1, -1], format='dok') + sp.sparse.diags(self.Vband[self.qind[0]:self.qind[1]], 0, format='dok')
        
        # Laplace for potential
        self.K0 = sp.sparse.diags([2*np.ones(self.npts), -np.ones(self.npts-1), -np.ones(self.npts-1)], [0, 1, -1], format='dok')
        self.K0 /= h**2
        
        self.beta = sp.constants.e/(sp.constants.k * T)
        self.dos = 1e-18 * meff * cn.m_e*cn.e/(np.pi*cn.hbar**2) 
        #(states per eV and nm^2)
    
        self.V = np.zeros(self.npts)

    def settemp(self, T):
        self.beta = sp.constants.e/(sp.constants.k * T)

    def setbias(self, V1, V2):
        self.Vbias = (V1, V2)
        
    def solve(self, varion, doquant, maxiter = 30, maxerr = 1e-10):  
              
        err = 1    
        i = 0
        self.nel = np.zeros(self.npts)

        while err > maxerr and i < maxiter:
          
            Vtot = self.V + self.Vband

            if varion:
                self.nion = self.nd/(1 + np.exp(-self.beta*(Vtot+self.Edx)))            
                dnion = sp.sparse.diags(\
                    self.beta*self.nd*np.exp(-self.beta*(Vtot+self.Edx))/(1 + np.exp(-self.beta*(Vtot+self.Edx)))**2, \
                    0, format='dok')
            else:
                dnion = 0.
            
            if doquant:
                            
                # no band energy in quantum region! (already included in setup)
                Q = (self.Q0 + sp.sparse.diags(self.V[self.qind[0]:self.qind[1]], 0, format='dok')).todia()
                
                self.E, self.psi = sp.sparse.linalg.eigsh(Q, which = 'SA', k = self.nev, ncv=20)

                # electron density in 1/nm^3
                
                self.p = (self.dos/self.beta)* np.log(np.exp(-self.E*self.beta)+1)
                self.nel[self.qind[0]:self.qind[1]] = (self.psi**2).dot(self.p) / self.h    
                # check max occupation
        
                # energy change + wave function change
        
                dEp = np.expand_dims(self.p, 0)*(1/(np.expand_dims(self.E, 0)-np.expand_dims(self.E, 1) + np.eye(self.nev))-np.eye(self.nev))
                A = np.expand_dims(self.psi, 0)*np.expand_dims(self.psi, 1) 
                
                 
        
                dnel = sp.sparse.dok_matrix((self.npts, self.npts))
                dnel[self.qind[0]:self.qind[1], self.qind[0]:self.qind[1]] = \
                -(self.dos/self.h) * ((self.psi**2)*(1/(np.exp(np.expand_dims(self.E, 0)*self.beta)+1))) \
                    .dot((self.psi**2).transpose()) \
                    + (2/self.h) * np.sum(A.dot(np.expand_dims(dEp, 0)).squeeze()*A, 2)
                
                #((psi*psi.transpose()/(E-E.transpose()))*psi).dot(log(exp(-E*beta)+1)
            else:
                self.nel =  1e-27 * (2 * self.meff * cn.m_e * cn.e * np.maximum(0, -(Vtot)) / cn.hbar**2)**1.5/(3*np.pi)
                dnel =  sp.sparse.diags(-1e-27 * (2 * self.meff * cn.m_e * cn.e * np.maximum(0, -(Vtot)) / cn.hbar**2)**.5/(2*np.pi)\
                    *(2 * self.meff * cn.m_e * cn.e * (0 > (Vtot)) / cn.hbar**2), 0, format='dok') 
                self.p = None                    
                self.E = None
                self.psi = None
            
            int_const = cn.e / (cn.epsilon_0 )*(self.nel - self.nion)*1e9
            rhs = np.multiply(np.divide(1,self.epsr_vec), int_const)
            # rhs = cn.e / (cn.epsilon_0 * self.epsr)*(self.nel - self.nion)*1e9
            rhs[0] += self.Vbias[0]/self.h**2
            rhs[-1] += self.Vbias[1]/self.h**2
           
            rf = self.K0.dot(self.V) - rhs
            df = (self.K0 - (cn.e / (cn.epsilon_0 * self.epsr))*(dnel - dnion)*1e9).tocsc() 

            self.V -= sp.sparse.linalg.spsolve(df, rf)
            err = np.abs(rf).max() # does not account for last update
            i += 1
            
        
        if i == maxiter:        
            print('WARNING: No convergence. err = %.1f after %d iterations.' % (err, i))
        # else:
        #     print('Convergence. err = %.1e after %d iterations.' % (err, i))

        #return V, nel, nion, p, E, psi

        

    def plotdensity(self, fig = 1, z1=0 , z2=100):

        plt.figure(fig, figsize=(6,5))
        plt.clf()
        plt.subplot(211)
        plt.plot(self.z, self.V+self.Vband, color='k')
        plt.xlim(z1, z2) 
        plt.xlabel('z (nm)')
        plt.ylabel('Band edge (eV)')

         
        plt.subplot(212)
        #plt.semilogy(z, nion, z, nel)
        plt.plot(self.z, self.nion*1e21, self.z, self.nel*1e21, self.z, self.nd*1e21) 
        plt.xlim(z1, z2)
        plt.ylabel('Carrier density (cm^-3)')
        plt.xlabel('z (nm)')
        plt.legend(['ionized donors', 'electrons', 'donors'])

    def plotdensity_v2(self, fig=1, z1=0, z2=100):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 5), sharex=True, dpi=100)
        fig.subplots_adjust(hspace=0.1)

        # Band edge plot
        ax1.plot(self.z, self.V + self.Vband, color='black', linewidth=1.5)
        ax1.set_xlim(z1, z2)
        ax1.set_ylabel("Band edge (eV)", fontsize=11)
        ax1.tick_params(labelsize=10)

        # Carrier density plot
        ax2.plot(self.z, self.nion * 1e21, label='ionized donors', linewidth=1.2)
        ax2.plot(self.z, self.nel * 1e21, label='electrons', linewidth=1.2)
        ax2.plot(self.z, self.nd * 1e21, label='donors', linewidth=1.2)
        ax2.set_xlim(z1, z2)
        ax2.set_xlabel("z (nm)", fontsize=11)
        ax2.set_ylabel("Carrier density (cm⁻³)", fontsize=11)
        ax2.tick_params(labelsize=10)
        ax2.legend(fontsize=9, loc='upper right', frameon=False)

        plt.tight_layout()
        return fig
   
    
    def plotstates(self, z1=0, z2=100, y1=-0.4, y2=0.4):
        fig, ax = plt.subplots()   
        # plt.subplot(111)
        ax.plot(self.z, self.V+self.Vband, color='k')
        ax.set_xlim(z1, z2) 
        ax.set_xlabel('z (nm)')
        ax.set_ylabel('Potential (eV)')
        ax.set_ylim(y1, y2)

    
        # plt.subplot(212)
        ax2 = ax.twinx()
        ax2.plot(self.z[self.qind[0]:self.qind[1]], np.abs(self.psi)**2)
        # plt.xlim(z1, z2) 
        # plt.xlabel('z (nm)')
        ax2.set_ylabel('Probability density')
        ax2.set_ylim(-0.1,0.1)
        ax2.legend(['ground state', 'first excited state', 'second excited state'], loc='lower right')
    
    def plotstates_v2(self, z1=0, z2=100, y1=-0.4, y2=0.4):
        fig, ax = plt.subplots(figsize=(4, 3.5), dpi=100)

        ax.plot(self.z, self.V + self.Vband, color='black', linewidth=1.5)
        ax.set_xlim(z1, z2)
        ax.set_ylim(y1, y2)
        ax.set_xlabel("z (nm)", fontsize=11)
        ax.set_ylabel("Potential (eV)", fontsize=11)
        ax.tick_params(labelsize=10)

        # Plot wavefunctions
        ax2 = ax.twinx()
        for i in range(min(3, self.psi.shape[1])):  # plot up to 3 states
            ax2.plot(self.z[self.qind[0]:self.qind[1]], np.abs(self.psi[:, i])**2, label=f"ψ{i}", linewidth=1.2)
        ax2.set_ylim(0, 0.1)
        ax2.set_ylabel("Probability density", fontsize=11)
        ax2.tick_params(labelsize=10)
        ax2.legend(fontsize=9, loc='upper right', frameon=False)

        plt.tight_layout()
        return fig


    def printresults(self):
        print ('----------------------------------------------')
        print ('2DEG density: %.2e cm^-2' % (self.p[0]*1e14))
        print ('Higher subbands: %.2e cm^-2' % (sum(self.p[1:])*1e14))
        print ('E_F %.2f meV' % (-1e3*self.E[0]))
        print ('----------------------------------------------')
   
        
    def coarse_run(self, Tinit = 300, T1 = 100, T2 = 20):       
        # generic run        

        print ('Coarse run at T = %.0f K...' % Tinit )
        self.settemp(Tinit) 
        self.solve(True, False, maxerr = 1e-4) 

        print ('Solving donor ionization at T = %.0f K...' % T1 )
        self.settemp(T1) 
        self.solve(True, False) 
       
        print ('Solving 2DEG at T = %.1f K...' % T2 )  
        self.settemp(T2) 
        self.solve(False, True) 

    def run(self, T):       
        # generic run        

        self.settemp(T)
        self.solve(False, True) 
    
    def voltage_sweep(self, V_gate_min: float, V_gate_max: float, T: float, num_pts=50):
        """
        V_gate_min: voltage sweep minimum
        V_gate_max: voltage sweep maximum
        T: temperature (K)
        num_pts: number of points in sweep
        """

        integrated_densities = []

        # Extract quantum region
        qr_start, qr_end = self.qind

        self.setbias(-V_gate_max, 0) 
        self.coarse_run()

        V_arr = np.linspace(-V_gate_max, -V_gate_min, num_pts)

        for v in V_arr:
            # Setup simulation
            self.setbias(v, 0)
            self.run(T)

            # Integrate electron density (cm^-2)
            integrated_density = 1e14*np.trapezoid(self.nel[qr_start:qr_end], x=self.z[qr_start:qr_end]) 
            integrated_densities.append(integrated_density)

        self.V_gate_arr = -V_arr
        self.sheet_densities = np.array(integrated_densities)
