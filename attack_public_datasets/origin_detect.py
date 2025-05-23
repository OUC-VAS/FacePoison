
import sys
sys.path.append('.')
from detectors import *
sys.path.append('./detectors/yolov5face')  # yolov5face
from utils.bbox import draw_bboxes
from utils.evaluation import Evaluation
from utils.data import WIDER
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='WIDER', help='WIDER')
parser.add_argument('--iou_threshold', type=float, default=0.5)
parser.add_argument('--eva_p', type=str, default='/media/hash/data/code/FacePoison/attack_public_datasets/origin_eva_result')
parser.add_argument('--origin_vis_dir', type=str, default='/media/hash/data/code/FacePoison/attack_public_datasets/save_data/wider/origin_vis/')
parser.add_argument('--img_max_w', type=int, default=10e5)
parser.add_argument('--img_max_h', type=int, default=10e5)
args = parser.parse_args()

detectors_list = ['RetinaFace', 'YOLOv5Face', 'PyramidBox', 'S3FD', 'DSFD']

# load images and detect
if args.dataset == 'WIDER':
    dataloader = WIDER(max_w=args.img_max_w, max_h=args.img_max_h)
 
eva = Evaluation(eval_path=args.eva_p + '.txt', iou_threshold=args.iou_threshold)

for detector in detectors_list:
    model = eval(detector)(device='cuda')
    for i in range(len(dataloader)):
        print('\r', detector, i, 'wider', end='', flush=True)
        data = dataloader[i]
        if data is None:
            continue
        img, boxes, img_name = data
        pred_boxes = model.detect_faces(img, conf_th=0.9, scales=[1])
        img = draw_bboxes(img, pred_boxes)
        name = '{}_{}.png'.format(detector, img_name)
        dataloader.save(img, os.path.join(args.origin_vis_dir, name))
        eva.update(boxes, pred_boxes)
    eva.save(desc='# {}, origin {}'.format(detector, 'wider'))
    eva.reset()
    del model
eva.finish()
   