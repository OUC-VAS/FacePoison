import time
import numpy as np
import cv2
import torch, copy
from collections import OrderedDict
from .nets import RetinaFaceNet
from .box_utils import PriorBox, decode, decode_landm, py_cpu_nms

PATH_WEIGHT = './detectors/retinaface/weights/mobilenet0.25_Final.pth'

cfg = {'backbone_name': 'mobilenet0.25',
            'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
            'in_channel': 32, 'out_channel': 64}

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True
class RetinaFace():

    def __init__(self, device='cuda'):

        tstamp = time.time()
        self.device = device

        print('[RetinaFace] loading with', self.device, cfg['backbone_name'])
        self.net = RetinaFaceNet(cfg=cfg).to(self.device)

        state_dict = torch.load(PATH_WEIGHT, map_location=self.device)
        if "state_dict" in state_dict.keys():
            new_state_dict = remove_prefix(state_dict['state_dict'], 'module.')
        else:
            new_state_dict = remove_prefix(state_dict, 'module.')
        check_keys(self.net, new_state_dict)
        self.net.load_state_dict(new_state_dict, strict=False)
        self.net.eval()
        print('[RetinaFace] finished loading (%.4f sec)' % (time.time() - tstamp))

    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        bboxes = np.empty(shape=(0, 5))

        for s in scales:
            if image.dtype == np.float32:
                image = copy.deepcopy(image.astype(np.uint8))

            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img = np.float32(image)

            img_width, img_height = img.shape[1], img.shape[0]
            # img -= (104, 117, 123)
            img -= (123, 117, 104)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(self.device)
            scale = torch.Tensor([img_width, img_height, img_width, img_height])
            scale = scale.to(self.device)

            loc, conf, landms = self.net(img)

            priorbox = PriorBox(image_size=(img_height, img_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data

            boxes = decode(loc.data.squeeze(0), prior_data, [0.1, 0.2])
            boxes = boxes * scale / s
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

            landms = decode_landm(landms.data.squeeze(0), prior_data, [0.1, 0.2])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1 / s
            landms = landms.cpu().numpy()

            inds = np.where(scores > conf_th)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            order = scores.argsort()[::-1]
            # order = scores.argsort()[::-1][:5000]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, 0.4)
            dets = dets[keep, :]
            landms = landms[keep]
            dets = np.concatenate((dets, landms), axis=1)

            for bbox in dets:
                bbox = bbox[:5]
                bboxes = np.vstack((bboxes, bbox))

        keep = py_cpu_nms(bboxes, 0.1)
        bboxes = bboxes[keep]

        return bboxes