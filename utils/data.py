"""
Load preprocessed wider.
Small faces are filtered out as they can hardly be used for face synthesis
"""
from cmath import inf
import os, pickle
from PIL import Image
import numpy as np

class WIDER(object):

    def __init__(self, max_w=10e5, max_h=10e5):
        self.image_dir = './attack_public_datasets/wider/imgs'
        self.img_list = sorted(os.listdir(self.image_dir))
        self.anno_dir = './attack_public_datasets/wider/annos'
        self.anno_list = os.listdir(self.anno_dir)
        self.max_w = max_w
        self.max_h = max_h

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        image_name = self.img_list[item].split('.')[0]
        img = Image.open(self.image_dir + '/' + image_name + '.png')
        w, h = img.size
        
        with open(self.anno_dir + '/' + image_name + '.p', 'rb') as f:
            bboxes, labels = pickle.load(f, encoding='latin1')

        # xywh to xyxy
        bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]

        # check if it needs resize
        if w > self.max_w or h > self.max_h:
            w_ratio = self.max_w / w
            h_ratio = self.max_h / h
            ratio = min(w_ratio, h_ratio)
            w_ = int(round(w * ratio))
            h_ = int(round(h * ratio))
            img = img.resize((w_, h_))
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * ratio
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * ratio
            bboxes = bboxes.astype(int)
        img = np.array(img)
        return img, bboxes, image_name

    def save(self, image, img_path):
        if not os.path.exists(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))
        Image.fromarray(np.uint8(image)).save(img_path)