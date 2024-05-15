import numpy as np
from scipy.stats import levy_stable
from cv2 import imread
from cv2 import COLOR_BGR2RGB
from cv2 import cvtColor
from .Saliency import *

class Cle(object):

	'''
	 Generates a scanpath computing eye movements as Levy flight on a saliency map
	
	- Description
	Generates a visual scanpath by computing gaze shifts as Levy flights on
	any kind of saliency map (bottom-up or top-down) computed for the 
	given image. Basically a simple, but slightly enhanced, implementation of the algorithm
	described in the original paper of  Boccignone & Ferraro [1].
	The only variant with respect to [1] is the use of an internal
	simulation step along which a number of candidate gaze shifts is
	sampled [2].

	- References
	[1] G. Boccignone and M. Ferraro, Modelling gaze shift as a constrained
		random walk, Physica A, vol. 331, no. 1, pp. 207-218, 2004.
	[2] G. Boccignone and M. Ferraro, Feed and fly control of visual 
		scanpaths for foveation image processing, Annals of telecommunications - 
		Annales des telecommunications 
		2012.
	'''
	
	def __init__(self, saliecyMap, tauV=0.01, numSampleLevy=50, dt=0.1, T=25):

		'''Initialize a CLE object.
		Arguments:
		saliencyMap: ndarray containing a saliency map (bottom-up or top-down)
		tauV: Damping parameter for the computation of the potential
		numSampleLevy: Number of samples as candidate fixation points from the alpha-stable distribution
		dt: Time step for the Euler discretization
		T: Temperature for Metropolis
		'''
		#self.salMap = saliecyMap
		self.tauV = tauV
		
		#Set the parameters of the alpha-stable distribution
		self.alpha_stable = 1	
		self.beta_stable = 0
		self.delta_stable = 0

		self.NUM_SAMPLE_LEVY = numSampleLevy
		self.mix = 0.5
		self.k_P = self.mix
		self.k_R = 1 - self.mix
		self.h = dt	
		self.T = T	

		self.rows = saliecyMap.shape[0]
		self.cols = saliecyMap.shape[1]
  
		# the sigma in the computation of delta_s_hat (saliency gain)
		self.foaSize = 1/6. * min([self.rows, self.cols])

		self.gamma_stable = (2*self.foaSize)**2

	def computePotential(self, sal):
		#V = np.exp(-self.tauV*self.salMap)*100.
		V = np.exp(-self.tauV*sal)*100.

		diffl = np.zeros((self.rows+2, self.cols+2))
		diffl[1:self.rows+1, 1:self.cols+1] = V

		# finite difference method per calcolare la derivata prima
		# del potenziale lungo gli assi x e y
		# questa operazione può essere semplificata utilizzando
		# np.diff sulla saliency map con n = 1 e specificando l'asse
		deltaN = diffl[0:self.rows, 1:self.cols+1] - V
		deltaS = diffl[2:self.rows+2, 1:self.cols+1] - V
		deltaE = diffl[1:self.rows+1, 2:self.cols+2] - V
		deltaW = diffl[1:self.rows+1, 0:self.cols] - V

		dV_x = (deltaW + deltaE) / 2
		dV_y = (deltaS + deltaN) / 2

		return dV_x, dV_y

	def sample_new_coordinates(self, x_old, y_old, gazeDir, dV_x, dV_y):
		#Distances drawn from the stable random number generator
		# campiono 50 distanze (NUM_SAMPLE_LEVY = 50) così da ottenere
		# 50 salti diversi da poter controllare direttamente
		r = levy_stable.rvs(alpha=self.alpha_stable, beta=self.beta_stable, loc=self.delta_stable, scale=self.gamma_stable, size=self.NUM_SAMPLE_LEVY)

		#Generate randomly a direction theta from a uniform distribution 
		#between  -pi and pi and as a function of previous direction
		theta = 2*np.pi*np.random.rand(self.NUM_SAMPLE_LEVY,) - np.pi + gazeDir

		dV_x = np.reshape(dV_x,-1, order='F')
		dV_y = np.reshape(dV_y,-1, order='F')

		# k_P e k_R sono semplicemente the weights nella combinazione del
		# voltaggio e di eta nella formula di Langevin
		#Compute  new gaze position of the FOA via Langevin equation
		x_new = np.round(x_old + self.h * (-self.k_P * dV_x[x_old] + self.k_R * np.multiply(r, np.cos(theta))))
		y_new = np.round(y_old + self.h * (-self.k_P * dV_y[y_old] + self.k_R * np.multiply(r, np.sin(theta))))

		return x_new, y_new

	# a jump is valid if it staies in the image
	def cleShiftGazeLevy(self, x_old, y_old, gazeDir, dV_x, dV_y, sal):

		validcord = [False]
		while not any(validcord):
			#Sample new gaze points
			x_new, y_new = self.sample_new_coordinates(x_old, y_old, gazeDir, dV_x, dV_y)
			#Verifies if the generated gaze shift is located within the image
			validcord = np.logical_and(np.logical_and(np.logical_and(x_new>=0, x_new<self.rows), y_new>=0), y_new<self.cols)

		#Retains only the valid ones
		x_new = x_new[validcord].astype(int)
		y_new = y_new[validcord].astype(int)

		varPhi = np.zeros(x_new.shape) #allocating
		for ww in range(len(x_new)):
			varPhi[ww] = sal[x_new[ww], y_new[ww]] - sal[x_old, y_old]

		# between all the jumps, store the the one that endup in the place with 
		# the biggest delta in saliency
		idxMax = np.argmax(varPhi)
		best_x_new = x_new[idxMax]
		best_y_new = y_new[idxMax]

		return best_x_new, best_y_new

	# saliency gain
	def cleWeightSal(self, x, y, sigma, sal):
		win = sigma//2
		# creo la finestra per il confronto tra
		# x_s e il suo vicino x, lo stesso per y_s
		xwin = np.arange(x-win, x+win).astype(int)
		ywin = np.arange(y-win, y+win).astype(int)
		# controllo che le finestre siano valide
		# (non escano dai contorni dell'immagine)
		valid = np.logical_and(np.logical_and(np.logical_and(xwin<0, xwin>self.rows), ywin<0), ywin>self.cols)
		valid = all(xwin>=0) and all(xwin<self.rows) and all(ywin>=0) and all(ywin<self.cols)

		if valid:
			X, Y = np.meshgrid(xwin, ywin)
			gauss_mask = np.exp(-(np.square(X-x) + np.square(Y-y)))
			sub_sal = sal[np.ix_(xwin, ywin)]

			F = np.multiply(sub_sal, gauss_mask)
			wsal = np.sum(F)
		
		else:
			wsal = sal[x,y]

		return wsal

	def generateScanpath(self, sal, numSteps, starting_point=None):
		
		# posizione della fixation iniziale nel centro della
		# saliency map
		if starting_point == None: 
			xc = self.rows//2
			yc = self.cols//2
		else:
			xc = starting_point[0]
			yc = starting_point[1]

		# storing new and old coords of the fixation
		x_old = xc
		y_old = yc
		x_new = xc
		y_new = yc
		# cerate gaze coordinates
		foaCord = np.array([x_new, y_new])
		oldDir = 0	

		maxsal = np.max(sal)
		minsal = np.min(sal)
		# min max scale of saliency map
		sal = 100 * np.divide((sal - minsal), (maxsal - minsal))

		# copmuting potential differences over the axis
		dV_x, dV_y = self.computePotential(sal)

		foaStore = []
		for i in range(numSteps):

			REJECTED = ACCEPTED_M = ACCEPTED_IMM = False

			# store gaze coordinates
			foaStore.append(foaCord)
			
			# produce and validate a jump
			x_new, y_new = self.cleShiftGazeLevy(x_old, y_old, oldDir, dV_x, dV_y, sal)

			sigma = self.foaSize
			w_sal_new = self.cleWeightSal(x_new, y_new, sigma, sal)
			w_sal_old = self.cleWeightSal(x_old, y_old, sigma, sal)

			deltaS = w_sal_new - w_sal_old 	#saliency gain  

			if deltaS <= 0:
				# se saliency gain è minore di 0 calcolo rho
				# valore per il metropolis step
				# e lo confronto col rapporto tra il saliency gain
				# e la temperatura del emtropolis algorithm
				#Metropolis Step
				p  = np.exp(deltaS/self.T)
				tr = min(p,1)

				rho = np.random.rand(1,1)

				if rho >= tr:
					#RANDOM SEARCH: REJECTED BY METROPOLIS!! KEEPING OLD FOA
					x_new = x_old
					y_new = y_old
					REJECTED=True
				#else RANDOM SEARCH: ACCEPTED BY METROPOLIS!!
				else:
					# se rho è minore del valore di soglia accetto la
					# la nuova posizione
					ACCEPTED_M = True
			#else RANDOM SEARCH: ACCEPTED IMMEDIATELY!!
			else:
				#se il saliency gain è maggiore di 0 allora accetto la
				# nuova posizione
				ACCEPTED_IMM = True

			#Computing the direction of flight
			isChangedPoint = (x_old != x_new) or (y_old != y_new)
			if isChangedPoint:
				xx = np.sqrt((x_old - x_new)**2) 
				yy = np.sqrt((y_old - y_new)**2)
				newDir = np.arctan(yy/(xx+np.finfo(float).eps))
			else:
				newDir = oldDir
			# the direction of the gaze is used to draw the  next theta

			oldDir = newDir
			
   			# aggiorno le coordinate
			x_old = x_new
			y_old = y_new
			# salvo le nuove coordinate del gaze
			foaCord = np.array([x_new, y_new])

		return np.vstack(foaStore)

class CLE:
    
    def __init__(self):
        self.img_path = ''
        self.img = np.array([])
        self.SM = np.array([])
        self.SM_fullSize = np.array([])
        self.scanpath = []
    
    def getScanPath(self, img_path, num_fixations = 10, saliency_type = '98', tauV=0.01, numSampleLevy=50, dt=0.1, T=25):
        self.img_path = img_path # image path
        img = imread(self.img_path) # load image
        self.img = cvtColor(img, COLOR_BGR2RGB) # convert image to RGB format
        if saliency_type == '98':
            saliency = IttiKoch98(self.img)
        else:
            saliency = IttiKoch01(self.img)
        self.SM = saliency.getSaliencyMap() # compute saliency maps and other maps
        self.SM_fullSize = saliency.getSaliencyResized()
        cle = Cle(saliecyMap = self.SM_fullSize, tauV = tauV, numSampleLevy = numSampleLevy, dt = dt, T = T)
        self.scanpath = cle.generateScanpath(sal = self.SM_fullSize, numSteps = num_fixations)
        return self.scanpath