import sys
sys.path.append('.')
from attacks.adv_attack import FacePoisonPP, BIM, CADVG
sys.path.append('../detectors/yolov5face')  # yolov5face
from detectors import *
from attack_public_datasets.adv_config_default import FacePoisonPP_cfg
import os, time, argparse
from PIL import Image
import numpy as np
import torch
from torch import nn
import pprint

# import random
# seed = 1234
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
# np.random.seed(seed)
# random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--feat_weight', type=str, default='0.2,0.3,0.5')
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--alpha', type=float, default=2)
parser.add_argument('--flip_sign_p', type=float, default=0.01)
parser.add_argument('--mask_p', type=float, default=0.9) # keep prob
parser.add_argument('--ens', type=int, default=30)
parser.add_argument('--n_iter', type=int, default=10)
parser.add_argument('--lam', type=float, default=1)
parser.add_argument('--term_weight', type=str, default='1,0')
parser.add_argument('--fdmean', type=float, default=1.)
parser.add_argument('--hybrid_mode', type=int, default=2)
parser.add_argument('--avgerage_grad', type=int, default=1)
parser.add_argument('--tag', type=str, default='test')
parser.add_argument('--image_dir', type=str,
                    default='/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/sample_img_data_5frame/')
parser.add_argument('--save_image_dir', type=str,
                    default='/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/adv_sample_img_data_5frame_test_time_cadvg/')
args = parser.parse_args()

# update parameters
common_cfg = {
    'feat_weight': args.feat_weight,
    'flip_sign_p': args.flip_sign_p,
    'eps': args.eps,
    'alpha': args.alpha,
    'mask_p': args.mask_p,
    'ens': args.ens,
    'n_iter': args.n_iter,
    'lambda': args.lam,
    'term_weight': args.term_weight,
    'fdmean': args.fdmean,
    'hybrid_mode': args.hybrid_mode,
    'avgerage_grad': args.avgerage_grad,
    'loss_norm': 'L2',
}
for k, v in FacePoisonPP_cfg.items():
    FacePoisonPP_cfg[k].update(common_cfg)
pprint.pprint(FacePoisonPP_cfg)

# attack_types = ['FacePoisonPP']
attack_types = ['CADVG']
detectors_list = [RetinaFace, YOLOv5Face, PyramidBox, S3FD, DSFD]
# detectors_list = [DSFD]       # test

image_dir = args.image_dir  # '/media/lyz/disk0/facepoison/deepfaker/data/sample_img_data/'

ids = ['id0', 'id1', 'id2']  # ['id0', 'id1', 'id2'] #, ['id3', 'id20', 'id27']
# ids = ['id4', 'id5', 'id6', 'id9', 'id10', 'id12', 'id16', 'id23']       # other

save_image_dir = args.save_image_dir  # '/media/lyz/disk0/facepoison/attack_Celeb_DF_scripts_journal/DF_data/adv_sample_img_data/'

start_time = time.time()

if torch.cuda.device_count() > 1:
    MULTI_GPU = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    device_ids = [0, 1]
else:
    MULTI_GPU = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if MULTI_GPU:
#     net = nn.DataParallel(net,device_ids=device_ids)
with open('detector_time_CADVG.txt', 'a') as f:       # for test time
    for detector in detectors_list:
        detector_start_time = time.time()
        model = detector(device='cuda')
        if MULTI_GPU:
            net = nn.DataParallel(model.net, device_ids=device_ids)
            # model.to(device)
            model.net = net.module

        for atk_n in range(len(attack_types)):
            atk_model = attack_types[atk_n]

            if detector.__name__.lower() == 'yolov5face':
                # eval(attack_types[atk_n] + '_cfg')['yolov5face']['transform'] = 'yolov5face'
                eval('FacePoisonPP_cfg')['yolov5face']['transform'] = 'yolov5face'      # BIM

            # c = eval(attack_types[atk_n] + '_cfg')[detector.__name__.lower()]
            c = eval('FacePoisonPP_cfg')[detector.__name__.lower()]     # BIM
            facepo = eval(atk_model)(model.net, c)
            for id in ids:
                names = sorted(os.listdir(os.path.join(image_dir, id)))
                eta = 0
                img_adv = 0
                for i, image_name in enumerate(names):
                    print('\r', detector.__name__, args.tag, i, len(names), end='', flush=True)
                    img = Image.open(os.path.join(image_dir, id, image_name))
                    img = np.array(img)
                    # img_adv = facepo.attack(img)
                    img_adv, eta = facepo.attack(img, image_name, eta, img_adv)  #cvavg

                    # random compare ffpp
                    # img_adv = facepo.random(img)
                    image_name = image_name.replace('jpg', 'png')
                    save_path = os.path.join(save_image_dir, id + '_' + detector.__name__ + '_' + args.tag, image_name)
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    Image.fromarray(np.uint8(img_adv)).save(save_path)
        del model
        detector_end_time = time.time() - detector_start_time
        print('\n[Time] Detector: {} | Time: {} min'.format(detector.__name__, detector_end_time / 60.))
        f.write('\n[Time] Detector: {} | Time: {} min'.format(detector.__name__, detector_end_time / 60.))
    # for atk_type in attack_types: 
    #     adv_cfg = eval(atk_type)
    #     if atk_type == 'FIA_multifeat':
    #         FeatAdv = FeatAdvMultiFeat
    #     feat_adv = FeatAdv(model.net, adv_cfg[detector.lower()])
    #     for id in ids:   
    #         names = sorted(os.listdir(os.path.join(image_dir, id)))
    #         for i, image_name in enumerate(names):
    #             print('\r', detector, atk_type, i, len(names), end='', flush=True)
    #             img = Image.open(os.path.join(image_dir, id, image_name))
    #             img = np.array(img)
    #             img_adv = feat_adv.attack(img, atk_type)
    #             image_name = image_name.replace('jpg', 'png')
    #             save_path = os.path.join(save_image_dir, id + '_' + detector + '_' + atk_type, image_name)
    #             if not os.path.exists(os.path.dirname(save_path)):
    #                 os.makedirs(os.path.dirname(save_path))
    #             Image.fromarray(np.uint8(img_adv)).save(save_path)    
    # del model

run_time = time.time() - start_time
print('Total time: {} min'.format(run_time / 60.))
