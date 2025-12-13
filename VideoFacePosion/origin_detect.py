import sys
sys.path.append('.')
sys.path.append('../detectors/yolov5face')  # yolov5face
from detectors import *
import numpy as np
from utils.bbox import draw_bboxes
from utils.evaluation import Evaluation
import argparse, os
from PIL import Image
import cv2
from optf import image_warp
from utils import proc_vid as pv
import math

parser = argparse.ArgumentParser()
parser.add_argument('--gt_detector', type=str, default='DSFD')      # test_self use second batter detector
parser.add_argument('--pred_detector', type=str, default='PyramidBox')
parser.add_argument('--iou_threshold', type=float, default=0.5)
parser.add_argument('--eva_p', type=str,
                    default='/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/adv_eva_result')
parser.add_argument('--image_dir', type=str,
                    default='/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/sample_img_data_5frame/')
parser.add_argument('--tag', type=str, default='test')
parser.add_argument('--exp', type=str, default='None')
args = parser.parse_args()

ids = ['id1', 'id2', 'id0']

gt_model = eval(args.gt_detector)(device='cuda')
pred_model = eval(args.pred_detector)(device='cuda')

def None_():
    atk_type = 'None'
    eva = Evaluation(eval_path=args.eva_p + '_' + atk_type + '_' + args.tag + '.txt',
                     iou_threshold=args.iou_threshold)
    vis_dir = args.eva_p + '_' + atk_type + '_' + args.tag + '111'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    for id in ids:
        clean_names = sorted(os.listdir(os.path.join(args.image_dir, id)))
        for i in range(len(clean_names)):
            print('\r', id, args.gt_detector, args.pred_detector, args.tag, atk_type, clean_names[i], i, len(clean_names), end='', flush=True)
            image_name = clean_names[i]
            clean_img = Image.open(os.path.join(args.image_dir, id, image_name))
            clean_img = np.array(clean_img)

            # gt_boxes
            if args.gt_detector == 'MTCNN':
                boxes, _ = gt_model.detect_faces(clean_img, conf_th=0.9, scales=[0.0625, 0.125, 0.25, 0.5, 1])
            elif args.gt_detector == 'FaceBoxes':
                boxes = gt_model.detect_faces(clean_img, conf_th=0.9, scales=[0.5, 1])
            else:
                boxes = gt_model.detect_faces(clean_img, conf_th=0.9, scales=[1])

            # pred_boxes
            if args.pred_detector == 'MTCNN':
                pred_boxes, _ = pred_model.detect_faces(clean_img, conf_th=0.9, scales=[0.0625, 0.125, 0.25, 0.5, 1])
            elif args.pred_detector == 'FaceBoxes':
                pred_boxes = pred_model.detect_faces(clean_img, conf_th=0.9, scales=[0.5, 1])
            else:
                pred_boxes = pred_model.detect_faces(clean_img, conf_th=0.9, scales=[1])

            gt_img_vis = draw_bboxes(clean_img, boxes)
            pred_img_vis = draw_bboxes(clean_img, pred_boxes)
            clean_img_vis = np.concatenate([gt_img_vis, pred_img_vis], 1)
            img_path = os.path.join(vis_dir, id + '_' + args.gt_detector + '_' + args.pred_detector + '_' + args.tag + '_' + image_name.replace('png', 'jpg'))
            Image.fromarray(np.uint8(clean_img_vis)).save(img_path)

            eva.update(boxes, pred_boxes)
    eva.save(desc='# gt_detector:{}, pred_detector:{}, attack:{}'.format(args.gt_detector, args.pred_detector, atk_type))
    eva.reset()
    eva.finish()
if __name__ == '__main__':
    if args.exp == 'None':
        None_()
