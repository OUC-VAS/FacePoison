import numpy as np
import cv2, copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
from .utils import dct_2d, idct_2d, letterbox, letterbox_, check_img_size
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import chain
from detectors import *
from utils.evaluation import IoU


class FacePoision(object):
    def __init__(self, net, params, device='cuda'):
        self.net = net
        self._feature = []
        self._grad = []
        self.device = device
        self.params = params

        self.layers = params['layer'].split(',')
        for layer in self.layers:
            eval('self.net.{}'.format(layer)).register_forward_hook(self._feature_hook())
            eval('self.net.{}'.format(layer)).register_backward_hook(self._grad_hook())
        
        self.loss_norm = params['loss_norm'] if 'loss_norm' in params.keys() else 'L1'

        if len(self.layers) > 1:
            self.feat_weight = [float(item) for item in params['feat_weight'].split(',')]
        else:
            self.feat_weight = [1]

    def _feature_hook(self):
        def hook(module, input, output):
            self._feature.append(output)
        return hook

    def _grad_hook(self):
        def hook(module, grad_input, grad_output):
            self._grad.append(grad_output[0])
        return hook

    def _transform(self, image):
        if self.params['transform'] == '-mean':
            img_mean = np.array([123., 117., 104.])[np.newaxis, np.newaxis, :].astype('float32')
            img_mean = torch.from_numpy(img_mean).cuda()
            image = image - img_mean
        if self.params['transform'] == 'norm':
            image = image / 255.
        if self.params['transform'] == 'norm,-mean':
            image = image / 255.
            img_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :].astype('float32')
            img_mean = torch.from_numpy(img_mean).cuda()
            img_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :].astype('float32')
            img_std = torch.from_numpy(img_std).cuda()
            image = (image - img_mean) / img_std

        if self.params['transform'] == 'yolov5face':
            img_size = 640
            imgsz = (640, 640)
            image = np.array(image.detach().cpu())

            img0 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img = letterbox(img0, new_shape=imgsz)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)

            im = copy.deepcopy(img)
            im0s = copy.deepcopy(img0)

            if len(im.shape) == 4:
                orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis=0)
            else:
                orgimg = im.transpose(1, 2, 0)

            orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
            img0 = copy.deepcopy(orgimg)
            h0, w0 = orgimg.shape[:2]
            r = img_size / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

            imgsz = check_img_size(img_size, s=self.net.stride.max())

            image = letterbox(img0, new_shape=imgsz)[0]
            image = torch.from_numpy(image).cuda()

        return image

    def _transform_yolov5(self, image):
        if self.params['transform'] == '-mean':
            img_mean = np.array([123., 117., 104.])[np.newaxis, np.newaxis, :].astype('float32')
            img_mean = torch.from_numpy(img_mean).cuda()
            image = image - img_mean
        if self.params['transform'] == 'norm':
            image = image / 255.
        if self.params['transform'] == 'norm,-mean':
            image = image / 255.
            img_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :].astype('float32')
            img_mean = torch.from_numpy(img_mean).cuda()
            img_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :].astype('float32')
            img_std = torch.from_numpy(img_std).cuda()
            image = (image - img_mean) / img_std

        if self.params['transform'] == 'yolov5face':

            img_size = 640
            imgsz = (640, 640)
            img = letterbox_(image, new_shape=imgsz, skip_else=False)[0]

            if len(img.shape) == 4:
                orgimg = np.squeeze(img.transpose(0, 2, 3, 1), axis=0)
            else:
                orgimg = img.permute(1, 2, 0)

            h0, w0 = orgimg.shape[:2]
            r = img_size / max(h0, w0)
            if r != 1:
                interp = 'area' if r < 1 else 'bilinear'
                if interp == 'bilinear':
                    orgimg = F.interpolate(orgimg.permute(2, 0, 1).unsqueeze(0), size=(int(h0 * r), int(w0 * r)), mode=interp, align_corners=False).squeeze(0)
                else:
                    orgimg = F.interpolate(orgimg.permute(2, 0, 1).unsqueeze(0), size=(int(h0 * r), int(w0 * r)), mode=interp).squeeze(0)
                orgimg = orgimg.permute(1, 2, 0)

            imgsz = check_img_size(img_size, s=self.net.stride.max())

            image = letterbox_(orgimg, new_shape=imgsz, skip_else=True)[0]

            image = image.float()
            image /= 255.0
        return image
    
    def forward(self, image):
        x = self._transform_yolov5(image)
        x = x.permute(2, 0, 1)
        x.unsqueeze_(0)
        self.net(x)

    def forward_yolov5(self, image):
        x = self._transform_yolov5(image)
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)
        self.net(x)

    def _compute_x_grad(self, x, org_pred):
        loss = self._loss(org_pred)
        self.net.zero_grad()
        loss.backward()
        # x.requires_grad = True
        grad = x.grad.data.clone()
        return grad

    def _loss(self, org_pred): 
        if self.loss_norm.lower() == 'l1':
            loss = nn.L1Loss()
        elif self.loss_norm.lower() == 'l2':
            loss = nn.MSELoss()
        elif self.loss_norm.lower() == 'kl':
            loss = nn.KLDivLoss()
        else:
            loss = nn.L1Loss()
        lv = 0
        for i in range(len(self.layers)):
            lv += self.feat_weight[i] * loss(org_pred[i], self._feature[i]) 
        return lv

    def attack(self, image):
        return image

# BIM
class BIM(FacePoision):
    def __init__(self, net, params, device='cuda'):
        super().__init__(net, params, device)

    def attack(self, image):
        eps = self.params['eps']
        alpha = self.params['alpha']
        n_iter = self.params['n_iter']

        org_img = torch.from_numpy(image).float().cuda()
        self.forward_yolov5(org_img)

        # original features 
        feature_origin = []
        for i in range(len(self.layers)):
            feature_origin.append(self._feature[i].clone().detach())

        self._feature = []
        eta = torch.empty_like(org_img).uniform_(-eps, eps)
        adv_img = torch.clamp(org_img + eta, 0., 255.).cuda()
        for i in range(n_iter):
            x_adv = adv_img.clone()
            x_adv.requires_grad = True
            self.forward_yolov5(x_adv)
            grad = self._compute_x_grad(x_adv, feature_origin)

            adv_img += alpha * torch.sign(grad)
            eta = torch.clamp(adv_img - org_img, -eps, eps)
            adv_img = torch.clamp(org_img + eta, 0.0, 255.)
            self._feature = []
            self._grad = []
        adv_img = np.array(adv_img.detach().cpu())
        return adv_img

# MIM
class MIM(FacePoision):
    def __init__(self, net, params, device='cuda'):
        super().__init__(net, params, device)

    def attack(self, image):
        eps = self.params['eps']
        alpha = self.params['alpha']
        lam = self.params['lambda']
        n_iter = self.params['n_iter']

        org_img = torch.from_numpy(image).float().cuda()
        self.forward_yolov5(org_img)
        # original features 
        feature_origin = []
        for i in range(len(self.layers)):
            feature_origin.append(self._feature[i].clone().detach())

        self._feature = []
        eta = torch.empty_like(org_img).uniform_(-eps, eps)
        adv_img = torch.clamp(org_img + eta, 0., 255.).cuda()
        g = 0      
        for _ in range(n_iter):
            x_adv = adv_img.clone()
            x_adv.requires_grad = True
            self.forward_yolov5(x_adv)
            grad = self._compute_x_grad(x_adv, feature_origin)
            g = lam * g + grad / torch.sum(torch.abs(grad))
            adv_img += alpha * torch.sign(g)
            eta = torch.clamp(adv_img - org_img, -eps, eps)
            adv_img = torch.clamp(org_img + eta, 0.0, 255.)
            self._feature = []
            self._grad = []
        adv_img = np.array(adv_img.detach().cpu())
        return adv_img

# DIM
class DIM(FacePoision):
    def __init__(self, net, params, device='cuda'):
        super().__init__(net, params, device)

    def attack(self, image):
        def input_diversity(x):
            h, w = x.shape[:2]
            rnd = torch.empty(1).uniform_(0.7, 0.9).item()

            rescaled = F.interpolate(x.permute(2, 0, 1).unsqueeze(0), size=(round(rnd*h), round(rnd*w)), mode='bilinear', align_corners=False).squeeze(0)
            h_rem = round(h * (1 - rnd))
            w_rem = round(w * (1 - rnd))
            pad_top = torch.randint(0, h_rem + 1, (1,)).item()
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(0, w_rem + 1, (1,)).item()
            pad_right = w_rem - pad_left
            padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            if np.random.uniform() > 0.5:
                return x
            else:
                return padded.permute(1, 2, 0)
                
        eps = self.params['eps']
        alpha = self.params['alpha']
        lam = self.params['lambda']
        n_iter = self.params['n_iter']

        org_img = torch.from_numpy(image).float().cuda()
        self.forward_yolov5(org_img)
        # original features 
        feature_origin = []
        for i in range(len(self.layers)):
            feature_origin.append(self._feature[i].clone().detach())

        self._feature = []
        eta = torch.empty_like(org_img).uniform_(-eps, eps)
        adv_img = torch.clamp(org_img + eta, 0., 255.).cuda()
        g = 0      
        for _ in range(n_iter):
            x_adv = adv_img.clone()
            x_ = input_diversity(x_adv)
            x_.requires_grad = True
            self.forward_yolov5(x_)
            grad = self._compute_x_grad(x_, feature_origin)
            g = lam * g + grad / torch.sum(torch.abs(grad))
            adv_img += alpha * torch.sign(g)
            eta = torch.clamp(adv_img - org_img, -eps, eps)
            adv_img = torch.clamp(org_img + eta, 0.0, 255.)
            self._feature = []
            self._grad = []
        adv_img = np.array(adv_img.detach().cpu())
        return adv_img

# NIM
class NIM(FacePoision):
    def __init__(self, net, params, device='cuda'):
        super().__init__(net, params, device)

    def attack(self, image):
        eps = self.params['eps']
        alpha = self.params['alpha']
        lam = self.params['lambda']
        n_iter = self.params['n_iter']

        org_img = torch.from_numpy(image).float().cuda()
        self.forward_yolov5(org_img)

        feature_origin = []
        for i in range(len(self.layers)):
            feature_origin.append(self._feature[i].clone().detach())

        self._feature = []
        eta = torch.empty_like(org_img).uniform_(-eps, eps)
        adv_img = torch.clamp(org_img + eta, 0., 255.).cuda()
        g = 0

        for _ in range(n_iter):
            x_adv = adv_img.clone()
            nes_img = x_adv + lam * alpha * g
            nes_img.requires_grad = True
            self.forward_yolov5(nes_img)
            grad = self._compute_x_grad(nes_img, feature_origin)
            g = lam * g + grad / torch.sum(torch.abs(grad))
            adv_img += alpha * torch.sign(g)
            eta = torch.clamp(adv_img - org_img, -eps, eps)
            adv_img = torch.clamp(org_img + eta, 0.0, 255.)
            self._feature = []
            self._grad = []
        adv_img = np.array(adv_img.detach().cpu())
        return adv_img

# FIA
class FIA(FacePoision):
    def __init__(self, net, params, device='cuda'):
        super().__init__(net, params, device)

    def _compute_x_grad(self, x, weight):
        loss = self._loss(weight)
        self.net.zero_grad()
        loss.backward()
        grad = x.grad.data.clone()
        # grad = grad.data.cpu().numpy().transpose((0, 2, 3, 1))[0]
        return grad

    def _loss(self, weight):
        lv = 0
        for i in range(len(self.layers)):
            lv += self.feat_weight[i] * torch.mean(weight[i] * self._feature[i])
        return lv

    def attack(self, image):
        eps = self.params['eps']
        alpha = self.params['alpha']
        lam = self.params['lambda']
        mask_p = self.params['mask_p']
        ens = self.params['ens']
        n_iter = self.params['n_iter']

        org_img = torch.from_numpy(image).float().cuda()
        self.forward(org_img)       
        # original features 
        feature_origin = []
        for i in range(len(self.layers)):
            feature_origin.append(self._feature[i].clone().detach())

        self._feature = []
        self._grad = []        
        weight = [[] for _ in range(len(self.layers))]  
        L1 = nn.L1Loss()
        for _ in range(ens):
            mask = torch.bernoulli(torch.ones_like(org_img) * mask_p)
            x_masked = org_img * mask
            self.forward(x_masked)                                       
            loss = L1(feature_origin[-1], self._feature[-1])
            self.net.zero_grad()
            loss.backward()
            for i in range(len(self.layers)):
                weight[i].append(self._grad[len(self.layers) - i - 1].detach().clone())
            self._feature = []
            self._grad = []

        for i in range(len(self.layers)): 
            weight[i] = torch.stack(weight[i], dim=0).sum(dim=0)
            weight[i] = weight[i] / weight[i].pow(2).sum(dim=(1,2,3)).sqrt()

        eta = torch.empty_like(org_img).uniform_(-eps, eps)
        adv_img = torch.clamp(org_img + eta, 0., 255.).cuda()
        g = 0      
        for _ in range(n_iter):
            x_adv = adv_img.clone()
            x_adv.requires_grad = True
            self.forward(x_adv)
            grad = self._compute_x_grad(x_adv, weight)
            g = lam * g + grad / torch.sum(torch.abs(grad))
            adv_img += alpha * torch.sign(g)
            eta = torch.clamp(adv_img - org_img, -eps, eps)
            adv_img = torch.clamp(org_img + eta, 0.0, 255.)
            self._feature = []
            self._grad = []
        adv_img = np.array(adv_img.detach().cpu())
        return adv_img


    def __init__(self, net, params, device='cuda'):
        super().__init__(net, params, device)

    def attack(self, image):
        eps = self.params['eps']
        alpha = self.params['alpha']
        lam = self.params['lambda']
        mask_p = self.params['mask_p']
        ens = self.params['ens']
        n_iter = self.params['n_iter']

        org_img = image.copy()
        self.forward(org_img)       
        # original features 
        feature_origin = []
        for i in range(len(self.layers)):
            feature_origin.append(self._feature[i].clone().detach())

        self._feature = []
        self._grad = []        
        weight = [[] for _ in range(len(self.layers))]  
        L1 = nn.L1Loss()
        for _ in range(ens):
            mask = np.random.binomial(1, mask_p, size=org_img.shape)
            x_masked = org_img * mask
            self.forward(x_masked)                                       
            loss = L1(feature_origin[-1], self._feature[-1])
            self.net.zero_grad()
            loss.backward()
            for i in range(len(self.layers)):
                weight[i].append(self._grad[len(self.layers) - i - 1].detach().clone())
            self._feature = []
            self._grad = []

        for i in range(len(self.layers)): 
            weight[i] = torch.stack(weight[i], dim=0).sum(dim=0)
            weight[i] = weight[i] / weight[i].pow(2).sum(dim=(1,2,3)).sqrt() 

        eta = np.random.uniform(-eps, eps, size=org_img.shape)
        adv_img = np.clip(org_img + eta, 0.0, 255.)  
        g = 0      
        for _ in range(n_iter):
            x_adv = self.forward(adv_img, is_grad=True) 
            grad = self._compute_x_grad(x_adv, weight)
            # more randomness: randomly flip the sign of grad
            flip_sign_p = self.params['flip_sign_p']
            sign_mask = np.random.binomial(1, flip_sign_p, size=org_img.shape)
            grad = grad * sign_mask + grad * (sign_mask - 1)            
            g = lam * g + grad / np.sum(np.abs(grad))
            adv_img += alpha * np.sign(g)
            eta = np.clip(adv_img - org_img, -eps, eps)
            adv_img = np.clip(org_img + eta, 0.0, 255.)
            self._feature = []
            self._grad = []
        return adv_img

# Our new method
class FacePoisonPP(FIA):   
    def __init__(self, net, params, device='cuda'):
        super().__init__(net, params, device)
        self.term_weight = [float(item) for item in params['term_weight'].split(',')]
        self.pre_grad = []
        
    def forward(self, image):
        x = self._transform(image)
        x_ = x.permute(2, 0, 1)
        x_.unsqueeze_(0)
        self.net(x_)

    def _loss(self, weight, feature_origin_for_loss):
        lv = 0
        for i in range(len(self.layers)):
            lv = lv + self.feat_weight[i] * torch.mean(weight[i] * self._feature[i])

        lv2 = self.add_distance_term(feature_origin_for_loss)
        total = self.term_weight[0] * lv + self.term_weight[1] * lv2
        # print('Loss: {}'.format(total))
        return total
    
    def _compute_x_grad(self, x, weight, feature_origin_for_loss):
        # x = x.data
        x.requires_grad = True

        self.forward_yolov5(x)
        # self.forward(x)
        loss = self._loss(weight, feature_origin_for_loss)
        self.net.zero_grad()
        loss.backward()
        grad = x.grad.data.clone()
        self._feature = []
        self._grad = []
        return grad


    def add_grad_sign_flip(self, grad):
        # more randomness: randomly flip the sign of grad
        flip_sign_p = self.params['flip_sign_p']
        # sign_mask = np.random.binomial(1, flip_sign_p, size=grad.shape)
        sign_mask = torch.bernoulli(torch.ones_like(grad) * flip_sign_p)
        grad = -grad * sign_mask + grad * (1 - sign_mask) 
        return grad           

    def add_distance_term(self, feature_origin):
        # add new feature distance metric
        # averaging features
        loss = nn.L1Loss()
        cos_sim = nn.CosineSimilarity(dim=0, eps=10e-6)
        # feat_dis = loss(self._feature[-1][0], feature_origin[0])
        feature_origin = torch.concat(feature_origin)
        # center_feat = torch.mean(feature_origin, dim=0)
        # center_dis = loss(self._feature[-1][0], center_feat)
        # standard deviations
        # dis = torch.empty(feature_origin.size()[0])
        # for i in range(len(feature_origin)):
        #     # dis[i] = loss(feature_origin[i], self._feature[-1][0])
        #     dis[i] = cos_sim(feature_origin[i].flatten(), self._feature[-1][0].flatten())
        # std_dis = torch.std(dis)
        # mean_dis = torch.mean(dis, dim=0)
        # lv2 = -mean_dis + std_dis
        sim = torch.empty(feature_origin.size()[0])
        dis = torch.empty(feature_origin.size()[0])
        for i in range(len(feature_origin)):
            dis[i] = loss(feature_origin[i], self._feature[-1][0])
            sim[i] = cos_sim(feature_origin[i].flatten(), self._feature[-1][0].flatten())
        std_sim = torch.std(sim)
        mean_sim = torch.mean(sim, dim=0)
        # std_dis = torch.std(dis)
        # mean_dis = torch.mean(dis)
        # lv2 = -center_dis + std_dis
        # lv2 = mean_sim + std_sim
        lv2 = -(self.params['fdmean'] * mean_sim + (1 - self.params['fdmean']) * std_sim)
        # lv2 = (self.params['fdmean'] * mean_dis + (1 - self.params['fdmean']) * std_dis)
        return lv2

    def spatial_attack(self, adv_img, org_img, weight, feature_origin_for_loss, g):
        eps = self.params['eps']
        alpha = self.params['alpha']
        lam = self.params['lambda']
        adv_x = adv_img.clone()

        grad = self._compute_x_grad(adv_x, weight, feature_origin_for_loss)

        self.pre_grad.append(grad)
        # check if using grad sign flip
        if self.params['flip_sign_p'] != 0:
            grad = self.add_grad_sign_flip(grad)

        if self.params['avgerage_grad'] == 1:
            grad = torch.stack(self.pre_grad, dim=0).mean(dim=0)
            # grad = (grad + self.pre_grad) / 2 if self.pre_grad is 0 else grad 
        # self.pre_grad = grad
        
        
        g = lam * g + grad / torch.sum(torch.abs(grad))
        adv_img = adv_img + alpha * torch.sign(g)
        eta = torch.clamp(adv_img - org_img, -eps, eps)
        adv_img = torch.clamp(org_img + eta, 0.0, 255.)
        # adv_img = adv_img.detach()
        return adv_img

    def freq_attack(self, adv_img, org_img, weight, feature_origin_for_loss, g):
        eps = self.params['eps']
        alpha = self.params['alpha']
        lam = self.params['lambda']

        N = 10 #20
        grad = 0
        sigma = 16 
        rho = 0.5  
        adv_x = adv_img.clone()
        adv_x = adv_x / 255.   
        adv_x = adv_x.permute(2, 0, 1)  
        for _ in range(N): # N = 20 in default                  
            gauss = torch.randn_like(adv_x) * (sigma / 255)
            gauss = gauss.cuda()
            x_dct = dct_2d(adv_x + gauss).cuda()
            mask = (torch.rand_like(x_dct) * 2 * rho + 1 - rho).cuda()
            x_idct = idct_2d(x_dct * mask)
            x_idct = x_idct * 255.
            x_idct = x_idct.permute(1, 2, 0)
            grad_ = self._compute_x_grad(x_idct, weight, feature_origin_for_loss)
            grad += grad_

        grad /= N
        self.pre_grad.append(grad)
        # check if using grad sign flip
        if self.params['flip_sign_p'] != 0:
            grad = self.add_grad_sign_flip(grad)

        if self.params['avgerage_grad'] == 1:
            grad = torch.stack(self.pre_grad, dim=0).mean(dim=0)
            # grad = (grad + self.pre_grad) / 2 if self.pre_grad is 0 else grad
        # self.pre_grad = grad
        
        g = lam * g + grad / torch.sum(torch.abs(grad))
        adv_img = adv_img + alpha * torch.sign(g)
        eta = torch.clamp(adv_img - org_img, -eps, eps)
        adv_img = torch.clamp(org_img + eta, 0.0, 255.)

        return adv_img
    def attack(self, image):
        eps = self.params['eps']        
        mask_p = self.params['mask_p']
        ens = self.params['ens']
        n_iter = self.params['n_iter']

        # arry to tensor
        org_img = torch.from_numpy(image).float().cuda()
        # org_img = image.copy()
        self.forward_yolov5(org_img)
        # original features
        feature_origin = []
        for i in range(len(self.layers)):
            feature_origin.append(self._feature[i].clone().detach())

        self._feature = []
        self._grad = []
        weight = [[] for _ in range(len(self.layers))]
        L1 = nn.L1Loss()
        feature_origin_for_loss = []
        feature_origin_for_loss.append(feature_origin[-1])

        for i in range(ens):
            mask = torch.bernoulli(torch.ones_like(org_img) * mask_p)
            x_masked = org_img * mask

            self.forward_yolov5(x_masked)
            loss = L1(feature_origin[-1], self._feature[-1])

            feature_origin_for_loss.append(self._feature[-1].clone().detach())
            self.net.zero_grad()
            loss.backward()
            for i in range(len(self.layers)):
                weight[i].append(self._grad[len(self.layers) - i - 1].detach().clone())
            self._feature = []
            self._grad = []

        for i in range(len(self.layers)):
            weight[i] = torch.stack(weight[i], dim=0).sum(dim=0)
            weight[i] = weight[i] / weight[i].pow(2).sum(dim=(1, 2, 3)).sqrt()

        eta = torch.empty_like(org_img).uniform_(-eps, eps)
        adv_img = torch.clamp(org_img + eta, 0., 255.).cuda()
        g = 0
        if self.params['hybrid_mode'] == 2:
            for i in range(n_iter):
                if i % 2 == 0:
                    # run spatial domain attack
                    adv_img = self.spatial_attack(adv_img, org_img, weight, feature_origin_for_loss, g)
                else:
                    # run frequency domain attack
                    adv_img = self.freq_attack(adv_img, org_img, weight, feature_origin_for_loss, g)

            self.pre_grad = []
        elif self.params['hybrid_mode'] == 1:
            for i in range(n_iter):
                adv_img = self.freq_attack(adv_img, org_img, weight, feature_origin_for_loss, g)
            self.pre_grad = []
        else:
            for i in range(n_iter):
                adv_img = self.spatial_attack(adv_img, org_img, weight, feature_origin_for_loss, g)
            self.pre_grad = []

        adv_img = np.array(adv_img.detach().cpu())
        # writer.close()
        return adv_img

    def random(self, image):
        eps = self.params['eps']
        org_img = torch.from_numpy(image).float().cuda()
        eta = torch.empty_like(org_img).uniform_(-eps, eps)
        adv_img = torch.clamp(org_img + eta, 0., 255.).cuda()
        adv_img = np.array(adv_img.detach().cpu())
        return adv_img
