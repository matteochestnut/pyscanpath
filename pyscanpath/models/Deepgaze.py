import torch
from .DeepGaze.deepgaze3 import DeepGazeIII
from cv2 import imread
from cv2 import COLOR_BGR2RGB
from cv2 import cvtColor
from cv2 import resize
try:
    import collections.abc
    collections.Sequence = collections.abc.Sequence
    collections.MutableMapping = collections.abc.MutableMapping
except ImportError:
    pass
from pysaliency.models import sample_from_logdensity
import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp

class Deepgaze:
    
    def __init__(self):
        self.img_path = ''
        self.img = np.array([])
        self.scanpath = []
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        torch.set_grad_enabled(False)
        self.model = DeepGazeIII(pretrained=True).to(self.device)
    
    def _get_fixation_history(self, fixation_coordinates):
        history = []
        for index in self.model.included_fixations:
            try:
                history.append(fixation_coordinates[index])
            except IndexError:
                history.append(np.nan)
        return history
    
    def getScanPath(self, img_path, num_fixations = 10):
        self.img_path = img_path
        img = imread(self.img_path)
        img_rgb = cvtColor(img, COLOR_BGR2RGB)
        self.img = resize(img_rgb, (1024, 768))
        #1,024 × 768
        
        centerbias_template = np.zeros((1024, 1024))
        centerbias = zoom(centerbias_template, (self.img.shape[0]/centerbias_template.shape[0], self.img.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
        centerbias -= logsumexp(centerbias)
        
        image_tensor = torch.tensor([self.img.transpose(2, 0, 1)]).to(self.device)
        centerbias_tensor = torch.tensor([centerbias]).to(self.device)
        
        fixations_x = [1024 // 2]
        fixations_y = [768 // 2]
        for i in range( num_fixations ):
                x_hist = self._get_fixation_history(fixations_x)
                y_hist = self._get_fixation_history(fixations_y)
                
                x_hist_tensor = torch.tensor([x_hist]).to(self.device)
                y_hist_tensor = torch.tensor([y_hist]).to(self.device)
                log_density_prediction = self.model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
                logD = log_density_prediction.detach().cpu().numpy()[0, 0]
                next_x, next_y = sample_from_logdensity(logD)

                fixations_x.append(next_x)
                fixations_y.append(next_y)
        
        fixations_x = (np.array(fixations_x) / 1024) * img.shape[1]
        fixations_y = (np.array(fixations_y) / 768) * img.shape[0]
        self.scanpath = np.array( [fixations_y[ 1 :].astype(int), fixations_x[ 1 :].astype(int)] ).transpose()
        return self.scanpath