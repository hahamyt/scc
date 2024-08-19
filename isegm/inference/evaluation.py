import os
from time import time

import cv2
import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker
from isegm.utils.vis import vis_result_base
from tqdm.contrib.concurrent import process_map
from functools import partial

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm

def get(dataset, predictor, pred_thr, max_iou_thr, min_clicks, max_clicks, callback, index):
    sample = dataset.get_sample(index)
    # save_dir = f'/root/workspace/xx/RSIS/experiments/vis_val/{predictor.name}/' + dataset.name + '/'
    # if not os.path.isdir(save_dir):
    #     # 创建文件夹
    #     os.makedirs(save_dir)
    for object_id in sample.objects_ids[:1]:
        _, sample_ious, _ = evaluate_sample(image=sample.image,
                                            gt_mask=sample.gt_mask(object_id),
                                            predictor=predictor,
                                            max_iou_thr=max_iou_thr,
                                            pred_thr=pred_thr,
                                            min_clicks=min_clicks,
                                            max_clicks=max_clicks,
                                            callback=callback)

    return sample_ious

def evaluate_dataset_mp(dataset, predictor, **kwargs):
    start_time = time()
    pred_thr = kwargs['pred_thr']
    max_iou_thr = kwargs['max_iou_thr']
    min_clicks = kwargs['min_clicks']
    max_clicks = kwargs['max_clicks']
    callback = None
    pfunc = partial(get, dataset, predictor, pred_thr, max_iou_thr, min_clicks, max_clicks, callback)
    all_ious = process_map(pfunc, list(range(len(dataset))), max_workers=2, chunksize=20, leave=False)
    end_time = time()
    elapsed_time = end_time - start_time
    return all_ious, elapsed_time

def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious = []

    save_dir = '../experiments/vis_val/SimpleClick/' + dataset.name + '/'
    if not os.path.isdir(save_dir):
        # 创建文件夹
        os.makedirs(save_dir)
    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        for object_id in sample.objects_ids[:1]:
            _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask(object_id), predictor,
                                                sample_id=index, save_dir=save_dir, **kwargs)
            all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time

@torch.cuda.amp.autocast(enabled=True)
def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None, save_dir=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            # import time
            # t = time.time()
            pred_probs, gcns = predictor.get_prediction(clicker)
            # t = t - time.time()
            # print(t)
            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            clicks_list = clicker.get_clicks()

            try:
                last_y, last_x = predictor.last_y, predictor.last_x
            except:
                last_x = -1
                last_y = -1

            # out_image = vis_result_base(image, pred_mask, gt_mask, iou, click_indx + 1, clicks_list,
            #                             np.zeros_like(gt_mask), last_y, last_x, gcns)
            # # if click_indx ==3 or click_indx == 0:
            # cv2.imwrite(
            #     save_dir + str(sample_id) + '_' + str(int(iou * 1000)) + '_' + str(click_indx + 1) + '.jpg',
            #     out_image)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
