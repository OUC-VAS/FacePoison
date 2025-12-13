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
parser.add_argument('--detector', type=str, default='DSFD')
parser.add_argument('--iou_threshold', type=float, default=0.5)
parser.add_argument('--eva_p', type=str,
                    default='/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/adv_eva_result111_cadvg')
parser.add_argument('--image_dir', type=str,
                    default='/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/sample_img_data_5frame/')
parser.add_argument('--adv_image_dir', type=str,
                    default='/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/adv_sample_img_data_5frame_test_time_cadvg/')
parser.add_argument('--tag', type=str, default='test')
parser.add_argument('--exp', type=str, default='origin')
args = parser.parse_args()

# image_dir = args.image_dir #'/media/lyz/disk0/facepoison/deepfaker/data/sample_img_data/'
ids = ['id1', 'id2', 'id0'] #['id0', 'id1', 'id2'] #, 'id3', 'id20', 'id27']
# ids = ['id1']       # test
start_index = {'id1': [0, 375, 650]}
# save_image_dir = args.save_image_dir #'/media/lyz/disk0/facepoison/attack_Celeb_DF_scripts_journal/DF_data/adv_sample_img_data/'
model = eval(args.detector)(device='cuda')

def origin():     # FP-all
    atk_type = 'origin'
    eva = Evaluation(eval_path=args.eva_p + '_' + atk_type + '_' + args.tag + '.txt',
                    iou_threshold=args.iou_threshold)
    vis_dir = args.eva_p + '_' + atk_type + '_' + args.tag + '111'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    for id in ids:   
        adv_names = sorted(os.listdir(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag)))
        clean_names = sorted(os.listdir(os.path.join(args.image_dir, id)))
        assert len(adv_names) == len(clean_names)
        for i in range(len(adv_names)):
            print('\r', id, args.detector, args.tag, atk_type, clean_names[i], i, len(adv_names), end='', flush=True)
            image_name = clean_names[i]
            clean_img = Image.open(os.path.join(args.image_dir, id, image_name))
            clean_img = np.array(clean_img)

            # if args.detector is 'MTCNN':
            if args.detector == 'MTCNN':
                boxes, _ = model.detect_faces(clean_img, conf_th=0.9, scales=[0.0625, 0.125, 0.25, 0.5, 1])
            # elif args.detector is 'FaceBoxes':
            elif args.detector == 'FaceBoxes':
                boxes = model.detect_faces(clean_img, conf_th=0.9, scales=[0.5, 1])
            else:
                boxes = model.detect_faces(clean_img, conf_th=0.9, scales=[1])

            image_name = adv_names[i]
            adv_img = Image.open(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag, image_name))
            adv_img = np.array(adv_img)

            if args.detector == 'MTCNN':
                pred_boxes, _ = model.detect_faces(adv_img, conf_th=0.9, scales=[0.0625, 0.125, 0.25, 0.5, 1])
            elif args.detector == 'FaceBoxes':
                pred_boxes = model.detect_faces(adv_img, conf_th=0.9, scales=[0.5, 1])
            else:
                pred_boxes = model.detect_faces(adv_img, conf_th=0.9, scales=[1])
            
            clean_img_vis = draw_bboxes(clean_img, boxes)
            adv_img_vis = draw_bboxes(adv_img, pred_boxes)
            img_vis = np.concatenate([clean_img_vis, adv_img_vis], 1)
            img_path = os.path.join(vis_dir, id + '_' + args.detector + '_' + args.tag + '_' + image_name.replace('png', 'jpg'))
            Image.fromarray(np.uint8(img_vis)).save(img_path)

            eva.update(boxes, pred_boxes)
    eva.save(desc='# detector:{}, attack:{}'.format(args.detector, atk_type))   
    eva.reset()
    eva.finish()
####################################################################################
####################################################################################
####################################################################################
def exp1():     # FP-fixed
    atk_type = 'exp1' # add adv of first frame to other frames
    eva = Evaluation(eval_path=args.eva_p + '_' + atk_type + '_' + args.tag + '.txt',
                    iou_threshold=args.iou_threshold)
    vis_dir = args.eva_p + '_' + atk_type + '_' + args.tag + '_'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    data_dir = '/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/test_video/'
    interval = 5
    for id in ids:   
        # number of videos    
        video_paths = sorted(os.listdir(os.path.join(data_dir, id)))
        n = 0
        n_ = 0
        start_index = []
        for dpath in video_paths:
            start_index.append(n_)
            dpath = os.path.join(data_dir, id, dpath)
            print(dpath)
            imgs_cur, frame_num, fps, width, height = pv.parse_vid(str(dpath))
            n += frame_num
            # n_ = math.ceil(n / 5)
            n_ += math.ceil(frame_num / 5)
        adv_names = sorted(os.listdir(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag)))
        clean_names = sorted(os.listdir(os.path.join(args.image_dir, id)))
        assert len(adv_names) == len(clean_names)
        cur_vid_index = 0
        for i in range(len(adv_names)):
            print('\r', id, args.detector, args.tag, atk_type, clean_names[i], cur_vid_index, i, len(adv_names), end='', flush=True)
            image_name = clean_names[i]
            clean_img = Image.open(os.path.join(args.image_dir, id, image_name))
            clean_img = np.array(clean_img, dtype=np.float32)

            if i in start_index: # check if another video starts
                # recalculate the noise
                cur_vid_index = 0
                image_name = adv_names[i]
                adv_img = Image.open(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag, image_name))
                adv_img = np.array(adv_img, dtype=np.float32)
                noise = adv_img - clean_img
            elif cur_vid_index % interval == 0: # interval to recalculate the noise
                image_name = adv_names[i]
                adv_img = Image.open(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag, image_name))
                adv_img = np.array(adv_img, dtype=np.float32)
                noise = adv_img - clean_img
            else:
                adv_img = clean_img + noise
                adv_img = np.clip(adv_img, 0, 255)

            if args.detector == 'MTCNN':
                boxes, _ = model.detect_faces(clean_img, conf_th=0.9, scales=[0.0625, 0.125, 0.25, 0.5, 1])
            elif args.detector == 'FaceBoxes':
                boxes = model.detect_faces(clean_img, conf_th=0.9, scales=[0.5, 1])
            else:
                boxes = model.detect_faces(clean_img, conf_th=0.9, scales=[1])

            if args.detector == 'MTCNN':
                pred_boxes, _ = model.detect_faces(adv_img, conf_th=0.9, scales=[0.0625, 0.125, 0.25, 0.5, 1])
            elif args.detector == 'FaceBoxes':
                pred_boxes = model.detect_faces(adv_img, conf_th=0.9, scales=[0.5, 1])
            else:
                pred_boxes = model.detect_faces(adv_img, conf_th=0.9, scales=[1])
            
            clean_img_vis = draw_bboxes(clean_img, boxes)
            adv_img_vis = draw_bboxes(adv_img, pred_boxes)
            img_vis = np.concatenate([clean_img_vis, adv_img_vis], 1)
            img_path = os.path.join(vis_dir, id + '_' + args.detector + '_' + args.tag + '_' + image_name.replace('png', 'jpg'))
            Image.fromarray(np.uint8(img_vis)).save(img_path)
            eva.update(boxes, pred_boxes)
            cur_vid_index += 1
    eva.save(desc='# detector:{}, attack:{}'.format(args.detector, atk_type))   
    eva.reset()
    eva.finish()

####################################################################################
####################################################################################
####################################################################################
def exp2():
    atk_type = 'exp2' # Optical flow
    eva = Evaluation(eval_path=args.eva_p + '_' + atk_type + '_' + args.tag + '.txt',
                    iou_threshold=args.iou_threshold)
    vis_dir = args.eva_p + '_' + atk_type + '_' + args.tag + '111'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    data_dir = '/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/test_video/'
    interval = 5
    for id in ids:   
        # number of videos    
        video_paths = sorted(os.listdir(os.path.join(data_dir, id)))
        n = 0
        n_ = 0
        start_index = []
        for dpath in video_paths:
            start_index.append(n_)
            dpath = os.path.join(data_dir, id, dpath)
            print(dpath)
            imgs_cur, frame_num, fps, width, height = pv.parse_vid(str(dpath))
            n += frame_num
            # n_ = math.ceil(n / 5)
            n_ += math.ceil(frame_num / 5)
            
        adv_names = sorted(os.listdir(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag)))
        clean_names = sorted(os.listdir(os.path.join(args.image_dir, id)))
        assert len(adv_names) == len(clean_names)
        cur_vid_index = 0
        for i in range(len(adv_names)):
            print('\r', id, args.detector, args.tag, atk_type, clean_names[i], cur_vid_index, i, len(adv_names), end='', flush=True)
            image_name = clean_names[i]
            clean_img = Image.open(os.path.join(args.image_dir, id, image_name))
            clean_img = np.array(clean_img, dtype=np.float32)

            if i in start_index: # check if another video starts
                # recalculate the noise
                cur_vid_index = 0
                image_name = adv_names[i]
                adv_img = Image.open(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag, image_name))
                adv_img = np.array(adv_img, dtype=np.float32)
                noise = adv_img - clean_img
            elif cur_vid_index % interval == 0: # interval to recalculate the noise
                image_name = adv_names[i]
                adv_img = Image.open(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag, image_name))
                adv_img = np.array(adv_img, dtype=np.float32)
                noise = adv_img - clean_img
            else:
                # Calculate optical flow
                clean_img_gray = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
                prev_clean_img_gray = cv2.cvtColor(prev_clean_img, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(clean_img_gray, prev_clean_img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                noise_warp = image_warp(noise, flow)
                noise_warp = np.clip(noise_warp, -8, 8)
                adv_img = clean_img + noise_warp
                # Image.fromarray(np.uint8(20*(noise_warp - noise))).save('test.jpg')
                noise = noise_warp
                adv_img = np.clip(adv_img, 0, 255)           

            prev_clean_img = clean_img
            if args.detector == 'MTCNN':
                boxes, _ = model.detect_faces(clean_img, conf_th=0.9, scales=[0.0625, 0.125, 0.25, 0.5, 1])
            elif args.detector == 'FaceBoxes':
                boxes = model.detect_faces(clean_img, conf_th=0.9, scales=[0.5, 1])
            else:
                boxes = model.detect_faces(clean_img, conf_th=0.9, scales=[1])

            if args.detector == 'MTCNN':
                pred_boxes, _ = model.detect_faces(adv_img, conf_th=0.9, scales=[0.0625, 0.125, 0.25, 0.5, 1])
            elif args.detector == 'FaceBoxes':
                pred_boxes = model.detect_faces(adv_img, conf_th=0.9, scales=[0.5, 1])
            else:
                pred_boxes = model.detect_faces(adv_img, conf_th=0.9, scales=[1])
            
            clean_img_vis = draw_bboxes(clean_img, boxes)
            adv_img_vis = draw_bboxes(adv_img, pred_boxes)
            img_vis = np.concatenate([clean_img_vis, adv_img_vis], 1)
            img_path = os.path.join(vis_dir, id + '_' + args.detector + '_' + args.tag + '_' + image_name.replace('png', 'jpg'))
            Image.fromarray(np.uint8(img_vis)).save(img_path)
            eva.update(boxes, pred_boxes)
            cur_vid_index += 1

    eva.save(desc='# detector:{}, attack:{}'.format(args.detector, atk_type))   
    eva.reset()
    eva.finish()

####################################################################################
####################################################################################
####################################################################################
def exp3():
    atk_type = 'exp3' # Optical flow + improve + forward mapping + backward mapping
    eva = Evaluation(eval_path=args.eva_p + '_' + atk_type + '_' + args.tag + '.txt',
                    iou_threshold=args.iou_threshold)
    vis_dir = args.eva_p + '_' + atk_type + '_' + args.tag + '111'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    data_dir = '/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/test_video/'
    interval = 5
    for id in ids:   
        # number of videos    
        video_paths = sorted(os.listdir(os.path.join(data_dir, id)))
        n = 0       # total_video_frame
        n_ = 0      # video and video segmentation
        start_index = []
        for dpath in video_paths:
            start_index.append(n_)
            dpath = os.path.join(data_dir, id, dpath)
            print(dpath)
            imgs_cur, frame_num, fps, width, height = pv.parse_vid(str(dpath))
            n += frame_num
            ## n_ = math.ceil(n / 5)    # ME
            n_ += math.ceil(frame_num / 5)
            
        adv_names = sorted(os.listdir(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag)))
        clean_names = sorted(os.listdir(os.path.join(args.image_dir, id)))
        assert len(adv_names) == len(clean_names)
        cur_vid_index = 0
        for i in range(len(adv_names)):
            print('\r', id, args.detector, args.tag, atk_type, clean_names[i], cur_vid_index, i, len(adv_names), end='', flush=True)
            image_name = clean_names[i]
            clean_img = Image.open(os.path.join(args.image_dir, id, image_name))
            clean_img = np.array(clean_img, dtype=np.float32)

            if i in start_index: # check if another video starts
                START = start_index.index(i)
                # recalculate the noise
                cur_vid_index = 0
                image_name = adv_names[i]
                adv_img = Image.open(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag, image_name))
                adv_img = np.array(adv_img, dtype=np.float32)
                noise = adv_img - clean_img
            elif cur_vid_index % interval == 0: # interval to recalculate the noise
                image_name = adv_names[i]
                adv_img = Image.open(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag, image_name))
                adv_img = np.array(adv_img, dtype=np.float32)
                noise = adv_img - clean_img
            else:
                # Calculate optical flow
                # forward mapping
                clean_img_gray = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
                prev_clean_img_gray = cv2.cvtColor(prev_clean_img, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(clean_img_gray, prev_clean_img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) # prev->current
                # visualize
                # import flow_vis
                # flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=True)
                # cv2.imwrite('op-test.jpg', flow_color)

                noise_warp = image_warp(noise, flow)
                # noise_warp = np.clip(noise_warp, -8, 8)
                # visualize
                # noise_warp = (noise_warp * 10) + 127
                # Image.fromarray(np.uint8(noise_warp)).show()

                # backward mapping
                if i+1 < len(adv_names):
                    if START + 1 >= len(start_index) or i+1 < start_index[START+1]:
                        next_image_name = clean_names[i+1]
                        next_clean_img = Image.open(os.path.join(args.image_dir, id, next_image_name))
                        next_clean_img = np.array(next_clean_img, dtype=np.float32)
                        next_clean_img_gray = cv2.cvtColor(next_clean_img, cv2.COLOR_RGB2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(clean_img_gray, next_clean_img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        # visualize
                        # import flow_vis
                        # flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=True)
                        # cv2.imwrite('op-test-re.jpg', flow_color)

                        next_image_name = adv_names[i+1]
                        next_adv_img = Image.open(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag, next_image_name))
                        next_adv_img = np.array(next_adv_img, dtype=np.float32)
                        next_noise = next_adv_img - next_clean_img
                        next_noise_warp = image_warp(next_noise, flow)
                        noise_warp = (noise_warp + next_noise_warp) / 2

                noise_warp = np.clip(noise_warp, -8, 8)
                # visualize
                # noise_warp = (noise_warp * 10) + 127
                # Image.fromarray(np.uint8(noise_warp)).show()

                adv_img = clean_img + noise_warp
                # Image.fromarray(np.uint8(20*(noise_warp - noise))).save('test.jpg')
                noise = noise_warp
                adv_img = np.clip(adv_img, 0, 255)           

            prev_clean_img = clean_img
            if args.detector == 'MTCNN':
                boxes, _ = model.detect_faces(clean_img, conf_th=0.9, scales=[0.0625, 0.125, 0.25, 0.5, 1])
            elif args.detector == 'FaceBoxes':
                boxes = model.detect_faces(clean_img, conf_th=0.9, scales=[0.5, 1])
            else:
                boxes = model.detect_faces(clean_img, conf_th=0.9, scales=[1])

            if args.detector == 'MTCNN':
                pred_boxes, _ = model.detect_faces(adv_img, conf_th=0.9, scales=[0.0625, 0.125, 0.25, 0.5, 1])
            elif args.detector == 'FaceBoxes':
                pred_boxes = model.detect_faces(adv_img, conf_th=0.9, scales=[0.5, 1])
            else:
                pred_boxes = model.detect_faces(adv_img, conf_th=0.9, scales=[1])
            
            clean_img_vis = draw_bboxes(clean_img, boxes)
            adv_img_vis = draw_bboxes(adv_img, pred_boxes)
            img_vis = np.concatenate([clean_img_vis, adv_img_vis], 1)
            img_path = os.path.join(vis_dir, id + '_' + args.detector + '_' + args.tag + '_' + image_name.replace('png', 'jpg'))
            Image.fromarray(np.uint8(img_vis)).save(img_path)
            eva.update(boxes, pred_boxes)
            cur_vid_index += 1

    eva.save(desc='# detector:{}, attack:{}'.format(args.detector, atk_type))   
    eva.reset()
    eva.finish()

####################################################################################
####################################################################################
####################################################################################
def exp4():
    atk_type = 'exp4' # Optical flow + Kalman filter
    eva = Evaluation(eval_path=args.eva_p + '_' + atk_type + '_' + args.tag + '.txt',
                    iou_threshold=args.iou_threshold)
    vis_dir = args.eva_p + '_' + atk_type + '_' + args.tag
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    data_dir = '/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/test_video/'
    interval = 5
    for id in ids:   
        # number of videos    
        video_paths = sorted(os.listdir(os.path.join(data_dir, id)))
        n = 0
        n_ = 0
        start_index = []
        for dpath in video_paths:
            start_index.append(n_)
            dpath = os.path.join(data_dir, id, dpath)
            print(dpath)
            imgs_cur, frame_num, fps, width, height = pv.parse_vid(str(dpath))
            n += frame_num
            ## n_ = math.ceil(n / 5)    # ME
            n_ += math.ceil(frame_num / 5)
            
        adv_names = sorted(os.listdir(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag)))
        clean_names = sorted(os.listdir(os.path.join(args.image_dir, id)))
        assert len(adv_names) == len(clean_names)
        cur_vid_index = 0
        for i in range(len(adv_names)):
            print('\r', id, args.detector, args.tag, atk_type, clean_names[i], cur_vid_index, i, len(adv_names), end='', flush=True)
            image_name = clean_names[i]
            clean_img = Image.open(os.path.join(args.image_dir, id, image_name))
            clean_img = np.array(clean_img, dtype=np.float32)

            if i in start_index: # check if another video starts
                # recalculate the noise
                cur_vid_index = 0
                image_name = adv_names[i]
                adv_img = Image.open(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag, image_name))
                adv_img = np.array(adv_img, dtype=np.float32)
                noise = adv_img - clean_img
            elif cur_vid_index % interval == 0: # interval to recalculate the noise
                image_name = adv_names[i]
                adv_img = Image.open(os.path.join(args.adv_image_dir, id + '_' + args.detector + '_' + args.tag, image_name))
                adv_img = np.array(adv_img, dtype=np.float32)
                noise = adv_img - clean_img
            else:
                # Calculate optical flow
                clean_img_gray = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
                prev_clean_img_gray = cv2.cvtColor(prev_clean_img, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(clean_img_gray, prev_clean_img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                noise_warp = image_warp(noise, flow)
                # noise = np.clip(noise, -8, 8)
                adv_img = clean_img + noise_warp
                # Image.fromarray(np.uint8(20*(noise_warp - noise))).save('test.jpg')
                noise = noise_warp
                adv_img = np.clip(adv_img, 0, 255)           

            prev_clean_img = clean_img
            if args.detector == 'MTCNN':
                boxes, _ = model.detect_faces(clean_img, conf_th=0.9, scales=[0.0625, 0.125, 0.25, 0.5, 1])
            elif args.detector == 'FaceBoxes':
                boxes = model.detect_faces(clean_img, conf_th=0.9, scales=[0.5, 1])
            else:
                boxes = model.detect_faces(clean_img, conf_th=0.9, scales=[1])

            if args.detector == 'MTCNN':
                pred_boxes, _ = model.detect_faces(adv_img, conf_th=0.9, scales=[0.0625, 0.125, 0.25, 0.5, 1])
            elif args.detector == 'FaceBoxes':
                pred_boxes = model.detect_faces(adv_img, conf_th=0.9, scales=[0.5, 1])
            else:
                pred_boxes = model.detect_faces(adv_img, conf_th=0.9, scales=[1])
            
            clean_img_vis = draw_bboxes(clean_img, boxes)
            adv_img_vis = draw_bboxes(adv_img, pred_boxes)
            img_vis = np.concatenate([clean_img_vis, adv_img_vis], 1)
            img_path = os.path.join(vis_dir, id + '_' + args.detector + '_' + args.tag + '_' + image_name.replace('png', 'jpg'))
            Image.fromarray(np.uint8(img_vis)).save(img_path)
            eva.update(boxes, pred_boxes)
            cur_vid_index += 1

    eva.save(desc='# detector:{}, attack:{}'.format(args.detector, atk_type))   
    eva.reset()
    eva.finish()

if __name__ == '__main__':
    if args.exp == 'origin':
        origin()
    if args.exp == 'exp1':
        exp1()
    if args.exp == 'exp2':
        exp2()
    if args.exp == 'exp3':
        import time
        detector_start_time = time.time()
        exp3()
        detector_end_time = time.time() - detector_start_time
        print('\n[Time] Detector: {} | Time: {} min'.format(args.detector, detector_end_time / 60.))
    if args.exp == 'exp4':
        exp4()
