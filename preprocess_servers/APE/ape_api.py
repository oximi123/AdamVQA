# Copyright (c) Facebook, Inc. and its affiliates.
import os
import pickle
from collections import abc
import cv2
import numpy as np
import tqdm
from PIL import Image
from detectron2.config import LazyConfig, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.utils.logger import setup_logger
from predictor_lazy import VisualizationDemo
from decord import VideoReader, cpu

import logging

logging.getLogger().setLevel(logging.ERROR)
import warnings

warnings.filterwarnings("ignore")

model2config = {
    'APE-L_D' : 'APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py',
    'APE-Ti' : 'APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitt_eva02_vlf_lsj1024_cp_16x4_1080k.py'
}

def setup_cfg(model_name):
    # load config from file and command-line arguments
    config_file = os.path.join(os.getenv('HOME'),
                               'code/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py')
    opts = ['train.init_checkpoint=' + os.path.join(os.getenv('HOME'), os.path.join('model/ape',model_name, 'model_final.pth')),
            'model.model_language.cache_dir=', 'model.model_vision.select_box_nums_for_evaluation=500',
            'model.model_vision.text_feature_bank_reset=True', 'model.model_vision.backbone.net.xattn=False']
    cfg = LazyConfig.load(config_file)
    cfg = LazyConfig.apply_overrides(cfg, opts)
    confidence_threshold = 0.1

    if "output_dir" in cfg.model:
        cfg.model.output_dir = cfg.train.output_dir
    if "model_vision" in cfg.model and "output_dir" in cfg.model.model_vision:
        cfg.model.model_vision.output_dir = cfg.train.output_dir
    if "train" in cfg.dataloader:
        if isinstance(cfg.dataloader.train, abc.MutableSequence):
            for i in range(len(cfg.dataloader.train)):
                if "output_dir" in cfg.dataloader.train[i].mapper:
                    cfg.dataloader.train[i].mapper.output_dir = cfg.train.output_dir
        else:
            if "output_dir" in cfg.dataloader.train.mapper:
                cfg.dataloader.train.mapper.output_dir = cfg.train.output_dir

    if "model_vision" in cfg.model:
        cfg.model.model_vision.test_score_thresh = confidence_threshold
    else:
        cfg.model.test_score_thresh = confidence_threshold

    setup_logger(name="ape")
    setup_logger(name="timm")

    return cfg


def ape_inference(input, text_prompt, demo):
    res_list = []
    res_list_path = '/home/ubuntu/code/APE/demo/tmp_file/res_list.pkl'

    for path in tqdm.tqdm(input):
        # use PIL, to be consistent with evaluation
        try:
            img = read_image(path, format="BGR")
        except Exception as e:
            continue

        predictions, visualized_output, visualized_outputs, metadata = demo.run_on_image(
            img,
            text_prompt=text_prompt,
            with_box=True,
            with_mask=False,
            with_sseg=False,
        )

        res = ""
        with_box = True
        if "instances" in predictions:
            results = instances_to_coco_json(
                predictions["instances"].to(demo.cpu_device), path
            )
            if with_box:
                for result in results:
                    res += metadata.thing_classes[result["category_id"]] + ": ["
                    for idx, box in enumerate(result['bbox']):
                        if idx != 3:
                            res += str(int(box)) + ", "
                        else:
                            res += str(int(box))
                    res += "]; "
            else:
                for result in results:
                    res += metadata.thing_classes[result["category_id"]] + ", "

        if len(res) > 0:
            if with_box:
                res_list.append(res[:-2])
            else:
                res_list.append(res)
        else:
            res_list.append("")
    pickle.dump(res_list, open(res_list_path, 'wb'))
    return res_list_path
