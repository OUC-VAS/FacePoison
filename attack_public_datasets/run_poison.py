import sys
sys.path.append('.')

from detectors import *
from attacks.adv_attack import FacePoisonPP, BIM, DIM, MIM, NIM
from utils.data import WIDER
from attack_public_datasets.adv_config_default import FacePoisonPP_cfg
import os, time, argparse
import pprint

sys.path.append('./detectors/yolov5face')  # yolov5face

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='WIDER', help='WIDER')
parser.add_argument('--adv_dir', type=str,
                    default='/media/hash/data/code/FacePoison/attack_public_datasets/save_data/wider/adv/')
parser.add_argument('--feat_weight', type=str, default='0.2, 0.3, 0.5')
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
parser.add_argument('--img_max_w', type=int, default=10e5)
parser.add_argument('--img_max_h', type=int, default=10e5)
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
}
for k, v in FacePoisonPP_cfg.items():
    FacePoisonPP_cfg[k].update(common_cfg)
pprint.pprint(FacePoisonPP_cfg)

attack_types = ['FacePoisonPP']
#attack_types = ['BIM', 'DIM', 'MIM', 'NIM']

detectors_list = [RetinaFace, YOLOv5Face, PyramidBox, S3FD, DSFD]

# load images and detect
if args.dataset == 'WIDER':
    dataloader = WIDER(max_w=args.img_max_w, max_h=args.img_max_h)

start_time = time.time()

for detector in detectors_list:
    model = detector(device='cuda')
    for atk_n in range(len(attack_types)):
        atk_model = attack_types[atk_n]

        if detector.__name__.lower() == 'yolov5face':
            eval('FacePoisonPP_cfg')['yolov5face']['transform'] = 'yolov5face'

        c = eval('FacePoisonPP_cfg')[detector.__name__.lower()]
        facepo = eval(atk_model)(model.net, c)
        for i in range(len(dataloader)):
            # print('\r', detector.__name__, atk_model, i, args.dataset, end='', flush=True)
            print('\r', detector.__name__, atk_model, i, args.dataset)
            data = dataloader[i]
            if data is None:
                continue
            img, _, img_name = data
            img_adv = facepo.attack(img)
            save_name = '{}_{}_{}.png'.format(detector.__name__, atk_model, img_name)
            dataloader.save(img_adv, os.path.join(args.adv_dir, save_name))           
    del model

run_time = time.time() - start_time
print('Total time: {} min'.format(run_time / 60.))
