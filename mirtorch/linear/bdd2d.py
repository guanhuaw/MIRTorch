import torch 
from .linearmaps import LinearMap
import numpy as np
from util import map2x, map2y, integrate1D, inter1d


class bdd(LinearMap):
    def __init__(self, size_in, size_out, **kwargs):
        super(bdd, self).__init__(size_in, size_out)
        # size_in: nx * ny
        # size_out: pSize * len(angle)
        self.DSD = kwargs['DSD']
        self.DS0 = kwargs['DS0'] 
        self.pSize = kwargs['pSize'] 
        self.dSize = kwargs['dSize']   
        self.nPix = kwargs['nPix']      
        self.nDet = kwargs['nDet']    
        self.angle = kwargs['angle']
        
        assert size_out[0] == len(self.angle)
        assert size_out[1] == self.nDet
        assert size_in[0] == self.nPix
        assert size_in[1] == self.nPix
        
        # Detector boundaries
        self.detX = torch.arange(-(self.nDet//2), (self.nDet//2)+1).mul(self.dSize)
        self.detY = (-(self.DSD-self.DS0)-(self.dSize/2)) * (torch.ones(self.nDet+1))

        # Pixel boundaries
        self.pixelX, self.pixelY = torch.meshgrid(torch.arange(-(self.nPix//2),(self.nPix//2)+1), 
                                    torch.arange(-(self.nPix//2),(self.nPix//2)+1), indexing='ij')
        self.pixelX = torch.transpose(self.pixelX, 0, 1) * self.pSize
        self.pixelY = torch.flipud(torch.transpose(self.pixelY, 0, 1) - 1/2)
        self.pixelY = self.pixelY * self.pSize
            
    def common(self, proj):
        beta = self.angle[proj] # angle from x-ray beam to y-axis
            
        # Tube rotation
        self.rtubeX = -self.DS0*torch.sin(beta)
        self.rtubeY = self.DS0*torch.cos(beta)

        # Detector rotation
        self.rdetX = self.detX*torch.cos(beta) - self.detY*torch.sin(beta)
        self.rdetY = self.detX*torch.sin(beta) + self.detY*torch.cos(beta)

        # Define angle case & which axis it project boundaries
        if (((beta >= np.pi/4) and (beta <= 3*np.pi/4)) or 
            ((beta >= 5*np.pi/4) and (beta <= 7*np.pi/4))):
            self.axisCase = False # map on y axis
        else:
            self.axisCase = True # map on x axis
        
        # Mapping boundaries onto a common axis
        if self.axisCase:
            self.detm = map2x(self.rtubeX,self.rtubeY,self.rdetX,self.rdetY)
            self.pixm = map2x(self.rtubeX,self.rtubeY,self.pixelX,self.pixelY)
            
        else:
            self.detm = map2y(self.rtubeX,self.rtubeY,self.rdetX,self.rdetY)
            self.pixm = torch.fliplr(torch.transpose(map2y(self.rtubeX,self.rtubeY, 
                                                           self.pixelX,self.pixelY), 0, 1))
        self.detSize = torch.diff(self.detm, dim = 0)

        self.L = torch.zeros(self.nDet)
            
        if self.axisCase: 
            for n in range(self.nDet):
                det_mid = (self.detm[n] + self.detm[n+1])/2
                theta = torch.atan(abs(self.rtubeX-det_mid)/abs(self.rtubeY)) # theta is the angle btw the ray to det_mid and y-axis
                self.L[n] = abs(self.pSize/((self.detSize[n]*torch.cos(theta))))
        else:
            for n in range(self.nDet):
                det_mid = (self.detm[n] + self.detm[n+1])/2
                theta = np.pi/2 - torch.atan(abs(self.rtubeY-det_mid)/abs(self.rtubeX))
                self.L[n] = abs(self.pSize/((self.detSize[n]*torch.sin(theta))))    

    def _apply(self, phantom):
        #### forward projection ######
        # For each projection
        sinogram = torch.zeros(len(self.angle), self.nDet)
        
        for proj in range(len(self.angle)):
            self.common(proj)
            if self.axisCase:
                img = phantom
            else:
                img = torch.fliplr(torch.transpose(phantom, 0, 1))
            
            sinoTmp = torch.zeros(self.nDet) # one "row" of sinogram
            
            # For each image row
            for row in range(self.nPix):   
                rowm = self.pixm[row,:] # mapped row from pixel mapped         
                pixSize = torch.diff(rowm, dim = 0)

                Ppj = integrate1D(img[row,:], pixSize)
            
                Pdk = inter1d(idx=rowm, val=Ppj, query=self.detm)
                sinoTmp = sinoTmp + abs(torch.diff(Pdk, dim = 0))          
            
            sinogram[proj,:] = sinoTmp.mul(self.L)
        
        return sinogram

    def _apply_adjoint(self, sinogram):
        #### backward projection ######
        reconImg = torch.zeros(self.nPix, self.nPix)
        for proj in range(len(self.angle)):
            reconImg_angle = torch.zeros(self.nPix, self.nPix)
            self.common(proj)
            for row in range(self.nPix):   
                rowm = self.pixm[row, :]
                Ppj = integrate1D(sinogram[proj,:].mul(self.L), self.detSize)
                assert len(self.detm) > 1
                if self.detm[0] > self.detm[1]:
                    Pdk = inter1d(idx=reversed(self.detm), 
                                  val=reversed(Ppj), 
                                  query=rowm)
                else:
                    Pdk = inter1d(idx=self.detm,
                                  val=Ppj,
                                  query=rowm)
                reconImg_angle[row, :] = abs(torch.diff(Pdk, dim = 0))
            if self.axisCase:     
                reconImg += reconImg_angle
            else:
                reconImg += torch.fliplr(reconImg_angle)
                
        return reconImg