import os
import torch

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from collections import Counter
from processing_inputs import prepare_image

def plot_PRCurve(recalls, precision, path, title):
  if not os.path.exists(path):
    os.makedirs(path)
  fig = plt.figure(figsize=(7,7))
  plt.plot(recalls, precision)
  plt.xlim(0)
  plt.ylim(0)
  plt.xlabel('Recall', size = 20)
  plt.ylabel("Precision", size = 20)
  plt.title(f'P-R for {title} class', size = 25)
  plt.savefig(f'{path}/{title}.jpg', bbox_inches='tight')

def IOU(boxes_preds, boxes_labels):
  # boxes_preds and boxes_labels are in shape (N,4) where N is number of boxes

  boxes_preds = boxes_preds.numpy()
  boxes_preds = torch.tensor(boxes_preds)
  boxes_labels = boxes_labels.numpy()
  boxes_labels = torch.tensor(boxes_labels)

  box1_x1 = boxes_preds[..., 0:1]
  box1_y1 = boxes_preds[..., 1:2]
  box1_x2 = boxes_preds[..., 2:3]
  box1_y2 = boxes_preds[..., 3:4]
  box2_x1 = boxes_preds[..., 0:1]
  box2_y1 = boxes_preds[..., 1:2]
  box2_x2 = boxes_preds[..., 2:3]
  box2_y2 = boxes_preds[..., 3:4]

  x1 = torch.max(box1_x1, box2_x1)
  y1 = torch.max(box1_y1, box2_y1)
  x2 = torch.max(box1_x2, box2_x2)
  y2 = torch.max(box1_y2, box2_y2)

  # .clamp(0) is for the case they aren't intersect (intersection = 0)
  intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

  box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
  box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

  return intersection / (box1_area + box2_area - intersection + 1e-6)



def average_precision(detections, gts, iou_threshold, class_):
    """
    Calculate the average precision for each class
    """
    epsilon = 1e-6
    keys_list = ['idx','box', 'class','score']
    # count the number of bboxes for each image in gts list
    amount_bboxes = Counter(gt[keys_list[0]] for gt in gts)
    amount_bboxes_tensor = {}
    # initializing a zeros tensor for each image with number of its bboxes
    for key, val in amount_bboxes.items():
        amount_bboxes_tensor[key] = torch.zeros(val)
    # sorting the prediction according to its scores
    detections.sort(key=lambda x : x[keys_list[3]], reverse = True)

    TP = torch.zeros((len(detections)))
    FP = torch.zeros((len(detections)))
    total_true_boxes = len(gts)

    # getting the ground truth boxes for the same detections image
    for detection_idx, detection in enumerate(detections):
        gt_boxes = [
            bbox for bbox in gts if bbox[keys_list[0]] == detection[keys_list[0]]
            ]
        # number of true bboxes for the image
        num_gts = len(gt_boxes)
        best_iou = 0

        # iterating over gt_boxes to get the best detected box
        # with the heighest IOU with the truth
        for idx, gt in enumerate(gt_boxes):
            iou = IOU(gt[keys_list[1]], detection[keys_list[1]])
            # print('iou',iou)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        # we will assign 1 for TP tensor for the box with the height IOU if
        # it is higher then the threshold
        if best_iou > iou_threshold:
            # we will first ensure that box isn't checked
            # befor and assign its value to 1 after checking
            if amount_bboxes_tensor[detection[keys_list[0]]][best_gt_idx] == 0:
                TP[detection_idx] = 1
                amount_bboxes_tensor[detection[keys_list[0]]][best_gt_idx] = 1 # now we have covered this bbox and will not back to it again
            else:
              FP[detection_idx] = 1
        else:
            FP[detection_idx] = 1

    TP_cumsum = torch.cumsum(TP, dim = 0)
    FP_cumsun = torch.cumsum(FP, dim = 0)
    recalls   = torch.divide(TP_cumsum, (total_true_boxes + epsilon))
    recalls   = torch.cat((torch.tensor([0]), recalls))
    precision = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsun + epsilon))
    precision = torch.cat((torch.tensor([1]), precision))

    if iou_threshold == 0.5:
      return [torch.trapz(precision, recalls), precision, recalls, class_]
    # calculate the area under PR graph
    return torch.trapz(precision, recalls)



def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold, num_classes, classes_map, path, keys_list = ['idx','box', 'class','score']
):
    """
    Calculating the mAP for all classes for one threshold

    Args:
    pred_boxes and true_boxes are list of bboxes the following format
    {
        idx: image index
        box : one box in xyxy format
        class: class index
        score: prediction score
    }
    """
    average_precisions = []
    reverse_dict = {v: k for k, v in classes_map.items()}
    all_detections = []
    all_gts = []
    # iterating over each class and get the matched boxes
    for c in range (num_classes):
        detections = [detection for detection in  pred_boxes if detection[keys_list[2]] == c]
        gts = [true_box for true_box in  true_boxes if true_box[keys_list[2]] == c]

        all_detections.append(detections)
        all_gts.append(gts)
    iou_threshold = [iou_threshold for i in range(num_classes)]
    classes = list(range(num_classes))
    average_precisions = list(map(average_precision, all_detections, all_gts, iou_threshold, classes))

    if iou_threshold[0] == 0.5:
      for _, precision, recalls, class_ in average_precisions:
        plot_PRCurve(recalls.numpy(), precision.numpy(), path, reverse_dict[class_])

      average_precisions = [average_precision[0] for average_precision in average_precisions ]
    # return the average of AP
    return sum(average_precisions) / len(average_precisions)

def handle_ds(ds,inference_model):
  pred_list = []
  gt_list   = []
  for i,sample in enumerate(ds):
      image = sample[0]
      input_image, ratio = prepare_image(image)
      detections = inference_model.predict(input_image)
      num_detections = detections.valid_detections[0]
      bboxes = detections.nmsed_boxes[0][:num_detections] / ratio
      classes = detections.nmsed_classes[0][:num_detections]
      scores = detections.nmsed_scores[0][:num_detections]
      # truth
      for bbox, _class in zip(sample[1], sample[2]):
        sample_dict = {
            'idx': i,
            'box':bbox * np.array([image.shape[1],image.shape[0],image.shape[1],image.shape[0]]),
            'class':_class,
        }
        gt_list.append(sample_dict)
      # prediction
      for bbox, _class, score in zip(bboxes, classes, scores):
        sample_dict = {
            'idx': i,
            'box':bbox,
            'class':_class,
            'score': score
        }
        pred_list.append(sample_dict)
  return pred_list, gt_list

def calculate_mAP(preddiction, truth, num_classes, classes_map, path, iou_thresholds: list, keys_list = ['idx','box', 'class','score']):
    """
    Calculating object detection model accuracy using mAP metric
    """
    mAP = []
    for iou_threshold in iou_thresholds:
        mAP.append(mean_average_precision(preddiction, truth, iou_threshold, num_classes, classes_map, path, keys_list))

    return tf.cast(sum(mAP) / len(mAP), tf.float32).numpy()