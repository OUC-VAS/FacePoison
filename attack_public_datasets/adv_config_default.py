
FacePoisonPP_common_cfg = {
    'feat_weight': '0.2,0.3,0.5',
    'flip_sign_p': 0.01,
    'transform': '-mean',     # (S3FD, DSFD, Pyramidbox, RetinaFace)
    # 'transform': 'yolov5face',
    'eps': 8,
    'alpha': 2,
    'mask_p': 0.4,
    'ens': 30,
    'n_iter': 10,
    'lambda': 1,
    'term_weight': '1,1', # loss term weight)
    'fdmean': 1,
    'hybrid_mode': 2, # 0: spatial, 1: freq, 2:both
    'avgerage_grad': 1, # i: use, 0: not use
}
FacePoisonPP_cfg = {        # attack_model_layer: num_use_[]; alpha_use_.
    'dsfd': {
        'layer': 'vgg[7],vgg[16],vgg[31]', 
        'feat_weight': '0.2,0.3,0.5',
    },
    's3fd': {
        'layer': 'vgg[7],vgg[16],vgg[31]',
        'feat_weight': '0.2,0.3,0.5',
    },
    'pyramidbox': {
        'layer': 'vgg[15],vgg[22],vgg[28]',
        'feat_weight': '0.2,0.3,0.5',
    },
    'retinaface': { # mobilenetv1
         'layer': 'body.stage1,body.stage2,body.stage3',
         'feat_weight': '0.2,0.3,0.5',
    },
    'yolov5face': {
        'layer': 'model[1],model[3],model[5]',
        'feat_weight': '0.2,0.3,0.5',
    }
}

for k, v in FacePoisonPP_cfg.items():
    FacePoisonPP_cfg[k].update(FacePoisonPP_common_cfg)
