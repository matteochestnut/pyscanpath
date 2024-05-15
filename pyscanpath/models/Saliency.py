from cv2 import pyrDown
from cv2 import cvtColor
from cv2 import COLOR_RGB2GRAY
from cv2 import absdiff
from cv2 import getGaborKernel
from cv2 import filter2D
from cv2 import resize
from cv2 import normalize
from cv2 import NORM_MINMAX
from cv2 import CV_32F
from cv2 import sepFilter2D
import math
import numpy as np
from abc import ABC, abstractmethod 
# from scipy.ndimage.filters import maximum_filter

class Saliency(ABC):
    '''
    Parent class Saliency, inherit from python abstract base class
    in orther to implement separately original and later version of
    Itti Koch saliency map
    
    # Attributes
    _NUM_LEVELS (int): numeber of levels in the pyramid image representation
    _MIN_LEVEL (int): minimum level for the center image in the center sorround computation
        level 0 is the original image
    _MIN_DELTA (int): minimum delta to select the sorround image in the center sorround computation
        (if center image is a level l, sorround image is l + delta)
    _MAX_DELTA (int): maximum delta to select the sorround image in the center sorround copmutation
    _COLOR_NORMALIZATION_THRESHOLD_RATIO (float): threshold to which pixels of the R, G, B channels
        are set to zero
    _GABOR_SIZE (int tuple): size of the kernel to perform Gabor filtering
    _GABOR_SIGMA
    _GABOR_WAVELENGTH
    _GABOR_ASPECT_RATIO
    _GABOR_PHASE
    _GABOR_ANGLES (float list): list of angles for which the orientation pyramid are computed
    '''
    
    _NUM_LEVELS = 9
    _MIN_LEVEL = 2
    _MAX_LEVEL = 4
    _MIN_DELTA = 3
    _MAX_DELTA = 4
    _COLOR_NORMALIZATION_THRESHOLD_RATIO = 0.1
    _GABOR_SIZE = (9, 9)
    _GABOR_SIGMA = 2.3
    _GABOR_WAVELENGTH = 7
    _GABOR_ASPECT_RATIO = 1
    _GABOR_PHASE = np.pi / 2
    _GABOR_ANGLES = [0, np.pi / 4, np.pi / 2, (3 / 4) * np.pi]
    
    def __init__(self, img):
        '''
        Initialize a Saliency object, called by subclasses IttiKochSaliency and IttiKoch01
        
        # Arguments
        img: RGB image
        '''
        self.img = img
        self.height = img.shape[0]
        self.width = img.shape[1]
        self._FM = []
        self._CM = []
        norm_img = normalize(img, None, 0, 1.0, NORM_MINMAX, dtype = CV_32F)
        self._SM = self._computeSaliency( norm_img )
    
    def _gaussianSubsample(self, img):
        '''
        Compute the Gaussian Pyramid Image from an image passed as argument
        '''
        layer = img.copy()
        pyramid = [layer]
        for _ in range(Saliency._NUM_LEVELS - 1):
            layer = pyrDown(layer)
            pyramid.append(layer)
        return pyramid
    
    @abstractmethod
    def _centerSorround(self, centerPyramid, sorroundPyramid):
        pass
    
    @abstractmethod
    def _maxNormalization(self, FM, normMax):
        pass

    def _sumNormMaps(self, normMaps, hegiht, width):
        '''
        Combine (just sum) the feature maps passed as a list.
        The maps are resized at the dimension of the 4th level of the pyramid image
        and summed together
        
        # Arguments
        normMaps: list of normalized feature maps to combine
        height: height of the image at the 4th level of the pyramid
        width: width of the image at the 4th level of the pyramid
        '''
        normSum = np.zeros( (hegiht, width) )
        for map in normMaps:
            # cv2.resize swap height and width
            normSum = normSum + resize(map, (width, hegiht) )
        return normSum
    
    @abstractmethod
    def _colorNormalization(self, r, g, b, I):
        pass
    
    def _makeOrientationPyramid(self, intensityPyramid, theta):
        '''
        Compute orientation pyramid by filtering layer from the intensity pyramid
        with a Gabor filter. 4 different angles are used, resulting in 4 different
        orientation pyramids. argument theta stores the angles.
        '''
        orientationPyramid = []
        gaborKernel = getGaborKernel(Saliency._GABOR_SIZE,
                                         Saliency._GABOR_SIGMA,
                                         theta,
                                         Saliency._GABOR_WAVELENGTH,
                                         Saliency._GABOR_ASPECT_RATIO,
                                         Saliency._GABOR_PHASE)
        for pyramid in intensityPyramid:
            orientationPyramid.append( filter2D(pyramid, -1, gaborKernel) )
        return orientationPyramid
    
    def _getDim(self, img):
        '''
        Return dimension of the image passed as argument
        '''
        return img.shape[0], img.shape[1]
    
    def _attenuateBorders(self, map, border_size):
        '''
        linearly attenuates a border region of borderSize
        on all sides of the 2d data array.
        '''
        result = np.copy(map)
        dsz = map.shape

        if border_size * 2 > dsz[0]:
            border_size = int(np.floor(dsz[0] / 2))
        if border_size * 2 > dsz[1]:
            border_size = int(np.floor(dsz[1] / 2))
        if border_size < 1:
            return result

        bs = np.arange(1, border_size + 1)
        coeffs = bs / (border_size + 1)

        # top and bottom
        rec = np.tile(coeffs[:, np.newaxis], (1, dsz[1]))
        result[bs - 1, :] *= rec
        range_ = dsz[0] - bs[::-1] + 1
        result[range_ - 1, :] *= rec

        # left and right
        rec = np.tile(coeffs[np.newaxis, :], (dsz[0], 1))
        result[:, bs - 1] *= rec
        range_ = dsz[1] - bs[::-1] + 1
        result[:, range_ - 1] *= rec
        
        return result
    
    def _makeIntensityConspicuity(self, img):
        '''
        Compute intensisty conspicuity map for the image passed
        - compute intensity
        - compute intensity pyramid image
        - compute intensity feature maps with center sorround
        - attenuate borders???????????????????????????????????????????
        - normalize the feature maps
        - combine (sum) the feature maps
        '''
        I = cvtColor(img, COLOR_RGB2GRAY)
        pyramid = self._gaussianSubsample(I)
        featureMaps = self._centerSorround(pyramid, pyramid)
        for map in featureMaps:
            map = self._attenuateBorders(map,  round(max(pyramid[4].shape) / 20))
        normMaps = []
        for map in featureMaps:
            normMaps.append( self._maxNormalization(map, 10) )
        self._FM.append(normMaps) # storing feature maps
        height, width = self._getDim(pyramid[4])
        intensityConspicuity = self._sumNormMaps(normMaps, height, width)
        return intensityConspicuity
    
    def _makeColorConspicuity(self, img):
        '''
        Compute color conspicuity map for the image passed
        - copmute r, g, b channels and the intensity image
        - clamp the r, g, b image based on intensity threshold
        - computed normalized color channel and compute red-green, green-red
            blue-yellow and yellow-blue
        - copmute normalized feature maps separately for red/green and blue/yellow channels
        - combine (sum) RG and BY maps together
        '''
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        I = cvtColor(img, COLOR_RGB2GRAY)
        # thresholding r, g, b
        r[I < (np.max(I) * Saliency._COLOR_NORMALIZATION_THRESHOLD_RATIO) ] = 0
        g[I < (np.max(I) * Saliency._COLOR_NORMALIZATION_THRESHOLD_RATIO) ] = 0
        b[I < (np.max(I) * Saliency._COLOR_NORMALIZATION_THRESHOLD_RATIO) ] = 0
        
        RG, GR, BY, YB = self._colorNormalization(r, g, b, I)
        
        pyramidRG = self._gaussianSubsample(RG)
        pyramidGR = self._gaussianSubsample(GR)
        featureMapsRG = self._centerSorround(pyramidRG, pyramidGR)
        for map in featureMapsRG:
            map = self._attenuateBorders(map,  round(max(pyramidRG[4].shape) / 20))
        normMapsRG = []
        height, width = self._getDim(pyramidRG[4])
        for map in featureMapsRG:
            normMapsRG.append( self._maxNormalization(map, 10) )
        self._FM.append(normMapsRG) # storing feature maps
        
        pyramidBY = self._gaussianSubsample(BY)
        pyramidYB = self._gaussianSubsample(YB)
        featureMapsBY = self._centerSorround(pyramidBY, pyramidYB)
        for map in featureMapsBY:
            map = self._attenuateBorders(map,  round(max(pyramidBY[4].shape) / 20))
        normMapsBY = []
        for map in featureMapsBY:
            normMapsBY.append( self._maxNormalization(map, 10) )
        self._FM.append(normMapsBY)
        
        colorConspicuity = []
        for i in range(len(normMapsRG)):
            colorConspicuity.append(normMapsRG[i] + normMapsBY[i])
        colorConspicuity = self._sumNormMaps(colorConspicuity, height, width)
        return colorConspicuity
    
    def _makeOrientationConspicuity(self, img):
        '''
        Compute orientation conspicuity map for image passed
        - compute intensity pyramid
        - compute an orientation pyramid for each angle using the intensity pyramid
        - compute the normalized feature maps for each pyramid and sum together the maps from the same pyramid
        - normalized the maps obtained from previous step
        - combine (sum) the normalized maps from previous step
        '''
        intensityPyramid = self._gaussianSubsample( cvtColor(img, COLOR_RGB2GRAY) )
        orientationPyramids = []
        height, width = self._getDim(intensityPyramid[4])
        orientationConspicuity = np.zeros((height, width))
        for theta in Saliency._GABOR_ANGLES:
            orientationPyramids.append( self._makeOrientationPyramid(intensityPyramid, theta) )
        for pyramid in orientationPyramids:
            featureMaps = self._centerSorround(pyramid, pyramid)
            for map in featureMaps:
                map = self._attenuateBorders(map,  round(max(orientationPyramids[0][4].shape) / 20))
            normMaps = []
            for map in featureMaps:
                normMaps.append( self._maxNormalization(map, 10) )
            self._FM.append(normMaps)
            sum = self._sumNormMaps(normMaps, height, width)
            norm = self._maxNormalization(sum, 10)
            orientationConspicuity += norm
        return orientationConspicuity
    
    def _computeSaliency(self, img):        
        self._CM.append( self._maxNormalization( self._makeIntensityConspicuity(img), 2 ) )
        self._CM.append( self._maxNormalization( self._makeColorConspicuity(img), 2 ) )
        self._CM.append( self._maxNormalization( self._makeOrientationConspicuity(img), 2 ) )
        saliency = (self._CM[0] + self._CM[1] + self._CM[2]) / 3
        return saliency
    
    def getSaliencyMap(self):
        return self._SM
    
    def getSaliencyResized(self):
        '''
        Return the saliency map scaled to the image orignal dimensions
        '''
        saliency = resize(self._SM, (self.width, self.height))
        return saliency
    
    def getConspicuityMaps(self):
        return np.array(self._CM)
    
    def getFeatureMaps(self):
        tmp = []
        for maps in self._FM:
            for map in maps:
                tmp.append( resize( map, (self._SM.shape[1], self._SM.shape[0]) ) )
        return np.array(tmp)

class IttiKoch98(Saliency):
    '''
    Compute Itti Koch saliency map as described in:
    "A model of saliency-based attention for rapid scene analisys"
    L. Itti, C. Kich, E. Niebur
    '''
    def __init__(self, img):
        Saliency.__init__(self, img)
        
    def _colorNormalization(self, r, g, b, I):
        '''
        Compute normalized r, g, b, y (yellow) color channels and return
        '''
        # normalized rgby channels
        R = r - (g + b) / 2; G = g - (r + b) / 2; B = b - (g + r) / 2; Y = ((r + g) / 2)  - (absdiff(r, g) / 2) - b
        # negative values set to 0
        R[R < 0] = 0; G[G < 0] = 0; B[B < 0] = 0; Y[Y < 0] = 0
        RG = R - G; GR = G - R; BY = B - Y; YB = Y - B
        return RG, GR, BY, YB
    
    def _centerSorround(self, centerPyramid, sorroundPyramid):
        '''
        Compute center sorround images.
        Given the center image at level i and the sorround image at level i + j, the coarse
        image is resized to the finer scale, then the difference between the two is computed
        '''
        featureMaps = []      
        for i in range(Saliency._MIN_LEVEL, Saliency._MAX_LEVEL + 1):
            for j in range(Saliency._MIN_DELTA, Saliency._MAX_DELTA + 1):
                dim1 = centerPyramid[i].shape[0]
                dim2 = centerPyramid[i].shape[1]
                scaled = resize( sorroundPyramid[i + j], (dim2, dim1) )
                featureMaps.append( absdiff(scaled, sorroundPyramid[i]) )
        return featureMaps
    
    # local max normalization
    def _maxNormalization(self, FM, normMax):
        '''
        Local max normalization.
        - input feature map is normalized between 0 and normMax. If normMax is 0 this
            step is skipped. Also a trehshold is copmuted
        - local maxima are found in the (normalized) feature map and kept according to the threshold
        - average of local maxima is copmuted
        - the feature map is normalized by the squared difference of normMax and
            the avrege of the local maxima
        '''
        FM[FM < 0] = 0
        if normMax != 0:
            FM = normalize(FM, None, 0, normMax, NORM_MINMAX, dtype = CV_32F)
            thresh = normMax / 10
        else:
            thresh = 1
        refFM = FM[1 : FM.shape[0] - 1, 1 : FM.shape[1] - 1]
        cond1 = np.logical_and( refFM >= FM[0 : FM.shape[0] - 2, 1 : FM.shape[1] - 1], refFM >= FM[2 : FM.shape[0], 1 : FM.shape[1] - 1])
        cond2 = np.logical_and( refFM >= FM[1 : FM.shape[0] - 1, 0 : FM.shape[1] - 2], refFM >= FM[1 : FM.shape[0] - 1, 2 : FM.shape[1]])
        cond3 = np.logical_and(cond1, cond2)
        localMaximaIndices = np.logical_and(cond3, refFM >= thresh)
        localMmaxima = refFM[localMaximaIndices]
        if localMmaxima.size == 0:
            return FM * ( (normMax)**2 )
        else:
            return FM * ( (normMax - np.nanmean(localMmaxima))**2 )
"""         maxima = maximum_filter(FM, size = (FM.shape[0] / 10, FM.shape[1] / 10))
        maxima = (FM == maxima)
        mnum = maxima.sum()
        maxima = np.multiply(maxima, FM)
        mbar = float(maxima.sum()) / mnum
        return FM * (normMax - mbar)**2 """

class IttiKoch01(Saliency):
    '''
    Compute Itti Koch saliency map as described in:
    "Feature combination strategies for saliency based visual attention systems"
    L. Itti, C. Kich
    With modifications in color normalization and center-sorround as reported in the SaliencyToolbox:
    https://github.com/DirkBWalther/SaliencyToolbox.git
    "Interactions of visual attention and object recognition: Computational modeling, algorithms, and psychophysics"
    D. Walter
    '''
    _ITERATIVE_MAX_NORMALIZATION_NUM_ITER = 3
    _ITERATIVE_MAX_NORMALIZATION_INHIBITION = 2.
    _ITERATIVE_MAX_NORMALIZATION_EXCITATION_CO = 0.5
    _ITERATIVE_MAX_NORMALIZATION_INHIBITION_CO = 1.5
    _ITERATIVE_MAX_NORMALIZATION_EXCITATION_SIG = 2.
    _ITERATIVE_MAX_NORMALIZATION_INHIBITION_SIG = 25.
    
    def __init__(self, img):
        Saliency.__init__(self, img)
        
    def _safeDevide(self, arg1, arg2):
        arg2[arg2 == 0] = 1
        result = np.divide(arg1, arg2)
        result[arg2 == 0] = 0
        return result
        
    def _colorNormalization(self, r, g, b ,I):
        """
        Color normalization as defined in the Saliency Toolbox
        """
        RG = self._safeDevide( (r - g), I )
        BY = self._safeDevide( (b - np.minimum(r, g)), I )
        return RG, RG, BY, BY
    
    def _centerSorround(self, centerPyramid, sorroundPyramid):
        """
        Center-sorround difference as implemented in the Saliency Toolbox.
        While computer the center-sorround the coarse scale image is not scaled
        to the fine image dimensions, instead all the images in the pyramid are
        scaled to the dimensions of the 4th layer
        """
        dim1 = sorroundPyramid[4].shape[0]
        dim2 = sorroundPyramid[4].shape[1]
        featureMaps = []      
        for i in range(Saliency._MIN_LEVEL, Saliency._MAX_LEVEL + 1):
            for j in range(Saliency._MIN_DELTA, Saliency._MAX_DELTA + 1):
                scaled1 = resize( centerPyramid[i + j], (dim2, dim1) )
                scaled2 = resize( sorroundPyramid[i], (dim2, dim1) )
                featureMaps.append( absdiff(scaled1, scaled2) )
        return featureMaps
    
    def _makeGaussianKernel(self, peak, sigma, maxHalfWidth):
        threshPercent = 1
        halfWidth = math.floor( sigma * math.sqrt(-2 * math.log(threshPercent / 100)) )
        if (maxHalfWidth > 0) and (halfWidth > maxHalfWidth):
            halfWidth = maxHalfWidth
        if peak == 0:
            peak = 1 / (sigma * math.sqrt(2 * math.pi))
        tmp = np.exp( ( -( np.array(range(1,halfWidth+1))**2 ) / (2 * sigma * sigma) ) ) / 10
        gaussKernel = np.concatenate([tmp[::-1], [peak], tmp])
        return gaussKernel
    
    # iterative max normalization
    def _maxNormalization(self, FM, normMax):
        """
        Max iterative normalization.
        First the image is normalized between 0 and normMax, if normMax is 0 then the image is not normalized.
        Then a loop processing starts where each iteration consists of self excitation and neighbor induced inhibition
        implemented by a "difference of Gaussians" filter, followed by rectification.
        """
        FM[FM < 0] = 0
        if normMax != 0:
            FM = normalize(FM, None, 0, normMax, NORM_MINMAX, dtype = CV_32F)
        size = max(FM.shape)
        maxHalfWidth = max( (math.floor(min(FM.shape) / 4) - 1), 0 )
        excitationSig = size * IttiKoch01._ITERATIVE_MAX_NORMALIZATION_EXCITATION_SIG * 0.01
        inhibitionSig = size * IttiKoch01._ITERATIVE_MAX_NORMALIZATION_INHIBITION_SIG * 0.01
        excitationPeak = IttiKoch01._ITERATIVE_MAX_NORMALIZATION_EXCITATION_CO / (excitationSig * math.sqrt(2 * math.pi))
        inhibitionPeak = IttiKoch01._ITERATIVE_MAX_NORMALIZATION_INHIBITION_CO / (inhibitionSig * math.sqrt(2 * math.pi))
        excitationKernel = self._makeGaussianKernel(excitationPeak, excitationSig, maxHalfWidth)
        inhibitionKernel = self._makeGaussianKernel(inhibitionPeak, inhibitionSig, maxHalfWidth)
        
        for _ in range(IttiKoch01._ITERATIVE_MAX_NORMALIZATION_NUM_ITER):
            excitation = sepFilter2D(FM, kernelX = excitationKernel, kernelY = excitationKernel, ddepth = -1)
            inhibition = sepFilter2D(FM, kernelX = inhibitionKernel, kernelY = inhibitionKernel, ddepth = -1)
            globalInhibition = 0.01 * IttiKoch01._ITERATIVE_MAX_NORMALIZATION_INHIBITION * FM.max()
            FM = FM + excitation - inhibition - globalInhibition
            FM[FM < 0] = 0
            
        return FM
        
        