import sys
sys.path.append('.')
from detectors import *
sys.path.append('./detectors/yolov5face')  # yolov5face
import numpy as np
from utils.bbox import draw_bboxes
from utils.evaluation import Evaluation
from utils.data import WIDER
import argparse, os, time
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='WIDER', help='WIDER')
parser.add_argument('--iou_threshold', type=float, default=0.5)
parser.add_argument('--eva_p', type=str,
                    default='/media/hash/data/code/face_poison/attack_public_datasets/adv_eva_result_adv')
parser.add_argument('--adv_dir', type=str,
                    default='/media/hash/data/code/face_poison/attack_public_datasets/save_data/wider/adv')
parser.add_argument('--adv_vis_dir', type=str,
                    default='/media/hash/data/code/face_poison/attack_public_datasets/save_data/wider/adv_vis')
parser.add_argument('--img_max_w', type=int, default=10e5)
parser.add_argument('--img_max_h', type=int, default=10e5)
args = parser.parse_args()


target_detectors_list = ['RetinaFace', 'YOLOv5Face', 'PyramidBox', 'S3FD', 'DSFD']
source_detectors_list = ['RetinaFace', 'YOLOv5Face', 'PyramidBox', 'S3FD', 'DSFD']

attack_types = ['FacePoisonPP']

# load images and detect
if args.dataset == 'WIDER':
    dataloader = WIDER(max_w=args.img_max_w, max_h=args.img_max_h)

eva_list = []
for atk_type in attack_types:    
    eva = Evaluation(eval_path=args.eva_p + '_' + atk_type + '.txt',
                 iou_threshold=args.iou_threshold)
    eva_list.append(eva)

start_time = time.time()

for target_detector in target_detectors_list:
    model = eval(target_detector)(device='cuda')
    for source_detector in source_detectors_list:
        for atk_n in range(len(attack_types)):
            atk_type = attack_types[atk_n]
            for i in range(len(dataloader)):
                print('\r', target_detector, source_detector, atk_type, i, args.dataset, end='', flush=True)
                data = dataloader[i]
                if data is None:
                    continue
                _, boxes, img_name = data
                # find adv image
                name = '{}_{}_{}.png'.format(source_detector, atk_type, img_name)
                img_adv_p = os.path.join(args.adv_dir, name)
                img_adv = np.array(Image.open(img_adv_p))
                pred_boxes = model.detect_faces(img_adv, conf_th=0.9, scales=[1])
                img_adv = draw_bboxes(img_adv, pred_boxes)
                name = '{}_{}_{}_{}.png'.format(target_detector, source_detector, atk_type, img_name)
                dataloader.save(img_adv, os.path.join(args.adv_vis_dir, name))
                eva_list[atk_n].update(boxes, pred_boxes)
            eva_list[atk_n].save(desc='# target:{}, source:{}, attack:{}, adv {}'.format(target_detector, source_detector, atk_type, args.dataset))   
            eva_list[atk_n].reset()
    del model

for i in range(len(attack_types)):    
    eva_list[i].finish()

run_time = time.time() - start_time
print('Total time: {} min'.format(run_time / 60.))
   