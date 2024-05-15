from cv2 import resize as resize
from cv2 import connectedComponents
from cv2 import erode
from cv2 import dilate
from cv2 import imread
from cv2 import COLOR_BGR2RGB
from cv2 import cvtColor
import numpy as np
import math
from .Saliency import *

class LIF:
    
    def __init__(self):
        self.timeStep = 0.0001
        self.Eleak = 0
        self.Eexc = 100 * pow(10, -3)
        self.Einh = -20 * pow(10, -3)
        self.Gleak = 1 * pow(10, -8)
        self.Gexc = 0
        self.Ginh = 0
        self.GinhDecay = 1
        self.Ginput = 5 * pow(10, -8)
        self.Vthresh = 0.001
        self.C = 1 * pow(10, -9)
        self.time = 0
        self.V = np.array([0])
        self.I = 0
        self.DoesFire = True
        
    def evolveLIF(self, time):
        dt = time - self.time
        
        # integrate
        self.V = self.V + (dt / self.C) * \
            (self.I - self.Gleak * (self.V - self.Eleak) - \
            self.Gexc * (self.V - self.Eexc) - \
            self.Ginh * (self.V - self. Einh))
        
        # clamp potentials
        self.V[self.V < self.Einh] = self.Einh
        
        # Ginh decay
        self.Ginh *= self.GinhDecay
        
        # track if any neuron fired
        if self.DoesFire:
            fire = np.array( np.ones(self.V.shape), dtype = bool )
        else:
            fire = np.array( np.zeros(self.V.shape), dtype = bool )
        spikes = np.logical_and( self.V > self.Vthresh, fire )
        
        # reset units that have fired
        self.V[spikes] = 0
        
        # update time
        self.time = time
        
        return spikes
        
class WTA:
    
    _IOR_DECAY = 0.9999
    _SM_OUTPUT_RANGE = 1 * pow(10, -9)
    _NOISE_AMP = 1 * pow(10, -17)
    _NOISE_CONST = 1 * pow(10, -14)
    
    
    def __init__(self, SM, CM = None, FM = None):
        self.sm = LIF()
        self.exc = LIF()
        self.inhib = LIF()
        self.SM = SM
        self.SMHeight = SM.shape[0]
        self.SMWidth = SM.shape[1]
        self.level4Height = math.ceil( self.SMHeight/ 16 )
        self.level4Width = math.ceil( self.SMWidth / 16 )
        self.SMlevel4 = resize(SM, (self.level4Width , self.level4Height )) # cv2 swap height and width
        self.CM = CM
        self.FM = FM
        
        self.sm.C = 5 * pow(10, -8)
        self.sm.Einh = 0
        self.sm.Eexc = 0
        self.sm.Gleak = 1 * pow(10, -7)
        self.sm.Ginh = np.zeros( ( self.level4Height, self.level4Width ) )
        self.sm.GinhDecay = WTA._IOR_DECAY
        self.sm.DoesFire = False
        self.sm.I = self.SMlevel4 * WTA._SM_OUTPUT_RANGE + \
            WTA._NOISE_AMP * np.random.random( (self.level4Height, self.level4Width) ) + \
            WTA._NOISE_CONST
        self.sm.V = np.zeros( (self.level4Height, self.level4Width) )
        self.exc.I = np.zeros( (self.level4Height, self.level4Width) )
        self.exc.V = np.zeros( (self.level4Height, self.level4Width) )
        self.exc.Ginh = 1 + pow(10, -2) 
        
        self.scanpath = [] 
    
    def _evolveWTA(self):
        '''
        Evolve winner takes all by one time step and returning the
        coordinates of the winning neuron
        '''
        time = self.exc.time + self.exc.timeStep
        winner = [-1, -1]
        
        # evolve sm
        _ = self.sm.evolveLIF(time)
        
        # set the input into the excitatory WTA neurons to the output of the sm
        self.exc.I = self.sm.V * self.exc.Ginput
        
        # evolve excitatory neurons
        spikes = self.exc.evolveLIF(time)
        
        # erase inhibition
        self.exc.Ginh = 0
        
        # did anyone fire?
        if spikes.any():
            # pixel coorinates of the spikes
            idx = np.nonzero(spikes)
            # linear coordinates of the spikes
            linear_idx = np.ravel_multi_index(idx, spikes.shape)
            # take the first spike
            idx = linear_idx[0]
            # convert linear coordinates of the first spike to pixel coordinates
            winner = np.unravel_index(idx, spikes.shape)
            # exciting inhibition interneuron
            self.inhib.Gexc = self.inhib.Gleak * 10
            #print('winner: ', winner, ', time: ', time * 1000)
        
        # evolve inhibitory neuron
        spike = self.inhib.evolveLIF(time)
        if spike.any():
            # trigger global inhibition
            self.exc.Ginh = 1 * pow(10, -2)
            # no need to be excited anymore
            self.inhib.Gexc = 0
        
        return winner
    
    def _estimateShape(self, winner):
        '''
        Estimate the shape around the winning position to
        trigger later the inhibition of return
        '''
        winMap = []
        winMap.append( self.SMlevel4 )
        winPos = winner
        
        # deecting most contribuiting maps to the winning position
        # in conspicuity and feature maps
        if self.CM is not None:
            idx = self.CM[:, winPos[0], winPos[1]].argmax()
            winMap.append(self.CM[idx, :, :])
        
        if self.FM is not None:
            idx = self.FM[:, winPos[0], winPos[1]].argmax()
            winMap.append(self.FM[idx, :, :])
        
        for i in range( len(winMap) ):
            # map segmentation
            binMap = self._segmentMap(winMap[i], winPos)
            
            # check if something has been actually been segmented
            newMap = np.array([])
            IORmask = np.array([])
            gotMap = False
            # reject segmented regions that are too big
            areaRatio = np.sum(binMap) / binMap.size
            if areaRatio > 0 and areaRatio < 0.1:
                # inhibition of return mask is the segmented region dilated
                # with a 2D circular structuring element
                IORmask = binMap
                se = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 1]])
                IORmask = dilate(IORmask.astype(np.uint8), se.astype(np.uint8), iterations = 1)
                #print('IOR mask: \n', IORmask)
                
                # erode the binary map
                tmp = erode(winMap[i].astype(np.uint8), se.astype(np.uint8), iterations = 1)
                # if after the erosion the winning position still holds a value then
                # the binary map eroded is the newMap, otherwise the original map is
                # eroded with a smaller se
                #print('eroded 1: \n', tmp)
                if tmp[winPos] > 0 and np.sum(tmp) > 0:
                    newMap = tmp
                else:
                    se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
                    tmp = erode(winMap[i].astype(np.uint8), se.astype(np.uint8), iterations = 1)
                    #print('eroded 2: \n', tmp)
                    if tmp[winPos] > 0 and np.sum(tmp) > 0:
                        newMap = tmp
                # if newMap holds values than it's labelled by connectivity 4
                # and labelled maps become the final map
                if newMap.size != 0 or newMap.any():
                    _, labels = connectedComponents(newMap.astype(np.uint8), connectivity = 4)
                    binMap = labels[labels == labels[winPos]]
                gotMap = True
                
                #print('final bin map: ', binMap)
                
                # only the first good winning map is considered
                break
                
            if gotMap == False:
                binMap = np.array([])
           
        return binMap, IORmask
    
    def _segmentMap(self, winMap, winPos):
        '''
        Segment saliency map (and conspicuity map and feature map)
        around the winning position.
        Returns a binary map.
        '''
        seedVal = winMap[winPos]
        thresh = 0.1 * seedVal
        # scaling the map by the value of the winning neuron and
        # threhsolding to obtaina black & white image
        bw = winMap / seedVal
        bw[bw < thresh] = 0
        bw[bw >= thresh] = 1
        #print('bw: \n', bw)
        # labelling the image based on connected components with connectivity 4
        _, labels = connectedComponents(bw.astype(np.uint8), connectivity = 4) # cv2.connectedComponents accepts uint8 only
        # if in the label map, the winning position has been labeled than the
        # result map is computed from the labels otherwise the resulting map is
        # the zeros map
        if labels[winPos] > 0:
            resultMap = np.array((labels == labels[winPos]), dtype = np.uint8)
        else:
            resultMap = np.zeros(winMap.shape)
        #print('segmented: \n', np.squeeze(resultMap))
        return np.squeeze(resultMap)
    
    def _applyIORmask(self, winner, shapeData, IORmask):
        '''
        Apply inhibition of return using IOR mask
        '''
        ampl = 0.1 * self.sm.V[winner]
        if shapeData.shape == self.sm.V.shape:
            binMap = IORmask
        else:
            binMap = resize(IORmask, (self.sm.V.shape[1], self.sm.V.shape[0]))
        self.sm.Ginh = self.sm.Ginh + ampl * binMap
    
    def _applyDiskIOR(self, winner):
        xx = np.array( [i for i in range(0, self.sm.V.shape[1])] ) - winner[1]
        yy = np.array( [i for i in range(0, self.sm.V.shape[0])] ) - winner[0]
        x, y = np.meshgrid(xx, yy)
        d = np.multiply(x, x) + np.multiply(y, y)
        pampl = 0.1 * self.sm.V[winner[0], winner[1]]
        mampl = 1 * pow(10, -4) * pampl
        psdev = 0.3 * (-1) / pow(2, 9)
        msdev = 4.0 * psdev
        g = pampl * np.exp( -0.5 * d / psdev**2 ) - mampl * np.exp( -0.5 * d / msdev**2 )
        self.sm.Ginh += g
    
    def generateScanpath(self, numFixations):
        numFixations
        self.scanpath = []
        
        for i in range(numFixations):
            winner = [-1, -1]
            
            # evolve WTA until there is a winner neuron
            while winner[0] == -1:
                winner = self._evolveWTA()
            
            shapeData, IORmask = self._estimateShape(winner)
            
            # trigger inhibition of return
            if shapeData.size == 0:
                # if the segmentation failed and no proto object is detected
                # to mask the winnig neuron neighborhood properly then the
                # classical disk shape inhibition is applied
                self._applyDiskIOR(winner)
            else:
                self._applyIORmask(winner, shapeData, IORmask)
            
            # converting winning neuron coordinates to image coordinates
            win2img = [winner[0] * pow(2, 4), winner[1] * pow(2, 4)]
            #print('winner2img: ', win2img)
            
            self.scanpath.append(win2img)
        
        self.scanpath = np.array(self.scanpath)
                
        return self.scanpath

class IttiKoch:
    
    def __init__(self):
        self.img_path = ''
        self.img = np.array([])
        self.SM = np.array([])
        self.SM_fullSize = np.array([])
        self.scanpath = []
    
    def getScanPath(self, img_path, num_fixations = 10, saliency_type = '01'):
        self.img_path = img_path # image path
        img = imread(self.img_path) # load image
        self.img = cvtColor(img, COLOR_BGR2RGB) # convert image to RGB format
        if saliency_type == '98':
            saliency = IttiKoch98(self.img)
        else:
            saliency = IttiKoch01(self.img)
        self.SM = saliency.getSaliencyMap() # compute saliency maps and other maps
        self.SM_fullSize = saliency.getSaliencyResized()
        CM = saliency.getConspicuityMaps()
        FM = saliency.getFeatureMaps()
        wta = WTA(self.SM_fullSize, CM, FM) # initializing winner takes all
        self.scanpath = wta.generateScanpath(numFixations = num_fixations) # fenerate scanpath
        return self.scanpath