
import numpy as np

def IoU(boxA, boxB):
    area_A = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    area_B = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    xx1 = np.maximum(boxA[0], boxB[0])
    yy1 = np.maximum(boxA[1], boxB[1])
    xx2 = np.minimum(boxA[2], boxB[2])
    yy2 = np.minimum(boxA[3], boxB[3])
    w_inter = np.maximum(0, xx2 - xx1 + 1)
    h_inter = np.maximum(0, yy2 - yy1 + 1)
    area_inter = w_inter * h_inter

    return area_inter / (area_A + area_B - area_inter)

class Evaluation():

    def __init__(self, eval_path, iou_threshold=0.5) -> None: 
        self.f = open(eval_path, mode='w')
        self.reset()       
        self.iou_threshold = iou_threshold  

    def reset(self):
        self.total_iou = 0
        self.total_recall = 0
        self.total_precision = 0
        self.total_f1score = 0  

        self.avg_iou = 0
        self.avg_recall = 0
        self.avg_precision = 0
        self.avg_f1score = 0

        self.count = 0

    def finish(self):
        self.f.close()

    def save(self, desc='Evaluation'):
        self.f.write('{}\n'.format(desc))
        self.f.write('@IoU:{}\n'.format(self.avg_iou))
        self.f.write('@Recall:{}\n'.format(self.avg_recall))
        self.f.write('@Precision:{}\n'.format(self.avg_precision))
        self.f.write('@F1-score:{}\n'.format(self.avg_f1score))
        self.f.write('@Count:{}\n'.format(self.count))
        self.f.flush()

    def update(self, gt_boxes, pred_boxes):
        gt_num = len(gt_boxes)
        pred_num = len(pred_boxes)
        img_iou = 0.0
        img_recall = 0.0
        img_precision = 0.0
        img_f1score = 0.0

        pred_dict = dict()

        for box in gt_boxes:
            max_iou = 0
            for i, pred_box in enumerate(pred_boxes):
                if i not in pred_dict.keys():
                    pred_dict[i] = 0
                iou = IoU(box, pred_box)
                if iou > max_iou:
                    max_iou = iou
                if iou > pred_dict[i]:
                    pred_dict[i] = iou
            img_iou += max_iou
        
        if gt_num * pred_num > 0:
            true_positive = 0.0
            for i in pred_dict.keys():
                if pred_dict[i] > self.iou_threshold:
                    true_positive += 1.0
            img_recall = true_positive / gt_num
            img_precision = true_positive / pred_num
            if img_recall * img_precision == 0:
                img_f1score = 0.0
            else:
                img_f1score = (2*img_recall*img_precision) / (img_recall+img_precision)
            img_iou = img_iou / gt_num
        self.total_iou += img_iou
        self.total_recall += img_recall
        self.total_precision += img_precision
        self.total_f1score += img_f1score
        self.count += 1

        self.avg_iou = self.total_iou / self.count
        self.avg_recall = self.total_recall / self.count
        self.avg_precision = self.total_precision / self.count
        self.avg_f1score = self.total_f1score / self.count

