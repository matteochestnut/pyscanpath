import torch
import numpy as np
from PIL import Image
from cv2 import imread
from cv2 import COLOR_BGR2RGB
from cv2 import cvtColor


from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from .Iorroi.imutils import pad_img_KAR, pad_array_KAR
#from .vis import draw_scanpath
from torchvision.transforms import Normalize, ToTensor, Compose
from .Iorroi.components import *
import torch.nn.functional as F
from .Iorroi.MDN import mixture_probability, sample_mdn

class IORROI:
    
    def __init__(self):
        self.img_path = ''
        self.img = np.array([])
        self.scanpath = []
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        torch.set_grad_enabled(False)
        
        # instantiating meta segment anything model
        MODEL_TYPE = "vit_h"
        CHECKPOINT_PATH = 'pyscanpath/models/Iorroi/checkpoint/SAM Vit H.pth'
        sam = sam_model_registry[MODEL_TYPE](checkpoint = CHECKPOINT_PATH)
        sam.to(device = self.device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
    
    def getScanPath(self, img_path, num_fixations = 10):
        self.img_path = img_path
        
        img_orig = Image.open( img_path )
        #print('img orig size: ', img_orig.size)
        imgs, (pad_w, pad_h) = pad_img_KAR(img_orig, 400, 300) # padding image
        ratio = imgs.size[0] / 400
        #print('img size: ', imgs.size)
        imgs = imgs.resize((400, 300)) # image resize
        #print('img resized size: ', imgs.size)
        
        image_bgr = imread(img_path)
        self.img = cvtColor(image_bgr, COLOR_BGR2RGB) # rgb image
        
        # set image dimensions
        transform = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        imgs = transform(imgs).unsqueeze(0)
        imgs = imgs.to( self.device )
        
        # extracting semantic region
        result = self.mask_generator.generate( self.img )
        
        # labeling segmentation regions to obtain a 2D integers map of the semantic features
        label = 1
        sem_infos = np.zeros( (self.img.shape[0], self.img.shape[1]) )
        for annotation in result:
            mask = annotation['segmentation']
            sem_infos[mask] = label
            label += 1
            
        # semantic feature processing
        sem_infos, (_, _) = pad_array_KAR(sem_infos, 300, 400) # padding
        sem_infos = torch.LongTensor(np.int32(sem_infos)).unsqueeze(0).unsqueeze(0) # adding dimensions
        sem_infos = sem_infos.to( self.device )
        fix_trans = torch.FloatTensor([0.19]).to( self.device )
        
        y, x = np.mgrid[0:300, 0:400]
        x_t = torch.from_numpy(x / 300.).float().reshape(1, 1, -1)
        y_t = torch.from_numpy(y / 300.).float().reshape(1, 1, -1)
        xy_t = torch.cat([x_t, y_t], dim=1).to( self.device )
        
        # -------------------------------------------------------------------------------------------------------
        # setting up fixations
        first_fix = first_fix_sampler.sample() # sample first fixations
        ob.set_last_fixation(first_fix[0], first_fix[1]) # oculomotor bias
        pred_sp_x = [first_fix[0]]
        pred_sp_y = [first_fix[1]]
        pred_sp_fd = list()
        
        # extracting features with VGG19
        feature = feature_extractor( imgs )
        
        # semantic feature processing
        sem_infos = F.interpolate(sem_infos.float(), size=[feature.size(2), feature.size(3)]).long()
        sem_features = torch.zeros((feature.size(0), 3001, feature.size(2), feature.size(3))).float().to(device)
        sem_features[0, ...].scatter_(0, sem_infos[0, ...], 1)
        
        # merge semantic features and features from VGG19
        fused_feature = fuser(feature, sem_features)
        
        state_size = [1, 512] + list(fused_feature.size()[2:])
        ior_state = (torch.zeros(state_size).to(device), torch.zeros(state_size).to(device))
        state_size = [1, 128] + list(fused_feature.size()[2:])
        roi_state = (torch.zeros(state_size).to(device), torch.zeros(state_size).to(device))
        
        pred_xt = torch.tensor(int(pred_sp_x[-1])).float().to(device)
        pred_yt = torch.tensor(int(pred_sp_y[-1])).float().to(device)
        roi_map = roi_gen.generate_roi(pred_xt, pred_yt).unsqueeze(0).unsqueeze(0)
        pred_fd = fix_duration(fused_feature, roi_state[0], roi_map)
        pred_sp_fd.append(pred_fd[0, -1].item() * 750)
        
        # cycling on number of fixations to predict
        #for step in range(0, num_fixations - 1):
        count = 0
        while True:
            ior_state, roi_state, _, roi_latent = iorroi_lstm(fused_feature, roi_map, pred_fd, fix_trans, ior_state, roi_state)

            mdn_input = roi_latent.reshape(1, -1)
            pi, mu, sigma, rho = mdn(mdn_input)

            pred_roi_maps = mixture_probability(pi, mu, sigma, rho, xy_t).reshape((-1, 1, 300, 400))
            samples = list()
            for _ in range(30):
                samples.append(sample_mdn(pi, mu, sigma, rho).data.cpu().numpy().squeeze())

            samples = np.array(samples)
            samples[:, 0] = samples[:, 0] * 300
            samples[:, 1] = samples[:, 1] * 300
            x_mask = (samples[:, 0] > 0) & (samples[:, 0] < 400)
            y_mask = (samples[:, 1] > 0) & (samples[:, 1] < 300)
            samples = samples[x_mask & y_mask, ...]
            
            sample_idx = -1
            max_prob = 0
            roi_prob = pred_roi_maps.data.cpu().numpy().squeeze()
            for idx, sample in enumerate(samples):
                sample = np.int32(sample)
                p_ob = ob.prob(sample[0], sample[1])
                p_roi = roi_prob[sample[1], sample[0]]
                if p_ob * p_roi > max_prob:
                    max_prob = p_ob * p_roi
                    sample_idx = idx
            
            if sample_idx == -1:
                fix = first_fix_sampler.sample()
                fix_x = fix[0] * ratio - pad_w // 2
                fix_y = fix[1] * ratio - pad_h // 2
                if (fix_x < 0 or fix_x > img_orig.size[0]) and (fix_y < 0 or fix_y > img_orig.size[1]):
                    break
                pred_sp_x.append(fix[0])
                pred_sp_y.append(fix[1])
            else:
                sample_x = samples[sample_idx][0]
                sample_y = samples[sample_idx][1]
                fix_x = sample_x * ratio - pad_w // 2
                fix_y = sample_y * ratio - pad_h // 2
                if (fix_x < 0 or fix_x > img_orig.size[0]) and (fix_y < 0 or fix_y > img_orig.size[1]):
                    break
                pred_sp_x.append(sample_x)
                pred_sp_y.append(sample_y)
                
            
            ob.set_last_fixation(pred_sp_x[-1], pred_sp_y[-1])
            
            pred_xt = torch.tensor(int(pred_sp_x[-1])).float().to(device)
            pred_yt = torch.tensor(int(pred_sp_y[-1])).float().to(device)
            roi_map = roi_gen.generate_roi(pred_xt, pred_yt).unsqueeze(0).unsqueeze(0)
            pred_fd = fix_duration(fused_feature, roi_state[0], roi_map)
            pred_sp_fd.append(pred_fd[0, -1].item() * 750)
            
            count += 1
            
            if count == num_fixations - 1:
                break
            
        #print('x pre: ', pred_sp_x)
        #print('y pre: ', pred_sp_y)
        pred_sp_x = [x * ratio - pad_w // 2 for x in pred_sp_x]
        pred_sp_y = [y * ratio - pad_h // 2 for y in pred_sp_y]
        #print('pad_w: ', pad_w)
        #print('pad_h: ', pad_h)
        #print('ratio: ', ratio)
        self.scanpaths = np.array( list( zip(pred_sp_y, pred_sp_x) ) )
        return self.scanpaths