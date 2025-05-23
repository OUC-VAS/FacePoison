import time
import numpy as np
import cv2, copy
import torch
# from .models import *
# from .utils import *
from .nets import YOLOv5FaceNet
from .box_utils import check_img_size, letterbox, non_max_suppression_face
from .box_utils import scale_coords, scale_coords_landmarks, xyxy2xywh


PATH_WEIGHT = './detectors/yolov5face/weights/yolov5n-0.5.pt'

class YOLOv5Face():

    def __init__(self, device='cuda'):

        tstamp = time.time()
        self.device = device

        print('[YOLOv5Face] loading with', self.device)
        self.net = YOLOv5FaceNet(PATH_WEIGHT, map_location=self.device)
        #
        for param in self.net.parameters():       # The code needs to be uncommented during attacks and commented out during inference.
             param.requires_grad = True           # 攻击时需要取消注释，推理时需要注释代码
        #
        print('[YOLOv5Face] finished loading (%.4f sec)' % (time.time() - tstamp))

    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        bboxes = np.empty(shape=(0, 5))

        img_size = 640
        conf_thres = 0.6
        iou_thres = 0.5
        imgsz = (640, 640)

        image = np.float32(image)

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img = letterbox(image, new_shape=imgsz)[0]
        # img = img[:, :, ::-1].transpose(2, 0, 1)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        im = copy.deepcopy(img)
        im0s = copy.deepcopy(image)

        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis=0)
        else:
            orgimg = im.transpose(1, 2, 0)

        # orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]
        r = img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=self.net.stride.max())

        img = letterbox(img0, new_shape=imgsz)[0]

        img = img.transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.net(img)[0]

        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):
            im0 = im0s.copy()

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()

                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    # conf = det[j, 4].detach().cpu().numpy()
                    landmark = det[j, 5:15].cpu().numpy()
                    x1 = int(xyxy[0])
                    y1 = int(xyxy[1])
                    x2 = int(xyxy[2])
                    y2 = int(xyxy[3])
                    score = conf
                    bbox = (x1, y1, x2, y2, score)
                    bboxes = np.vstack((bboxes, bbox))

        return bboxes