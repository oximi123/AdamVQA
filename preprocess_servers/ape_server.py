import socket
import pickle
import time
import os

import cv2
import numpy as np

from ape_tools.ape_api import setup_cfg
from preprocess_config import PrepConfig
from preprocess_servers.APE.predictor_lazy import VisualizationDemo
from util import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

modality = 'object'

from PIL import Image

def cold_start():
    pass
def save_frames(frames):
    file_paths = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        home_path = os.getenv('HOME')
        file_path = f'{home_path}/code/Video-RAG-master-main/vidrag_pipeline/restore/frame_{i}.png'
        img.save(file_path)
        file_paths.append(file_path)
    return file_paths


import tqdm
from detectron2.data.detection_utils import read_image
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
import matplotlib.pyplot as plt

# model2config = {
#     'APE-L_D': 'code/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py',
#     'APE-Ti': 'code/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitt_eva02_vlf_lsj1024_cp_16x4_1080k.py'
# }
from omegaconf import OmegaConf

model2config = OmegaConf.load('../ape.yaml').model2config


def check_config_consistency(frame_config, object_config):
    assert frame_config['num_frame'] == object_config['num_frame']
    assert frame_config['sampling_method'] == object_config['sampling_method']


default_score_threshold = 0.2
model2demo = {}
for model_name, model_config in model2config.items():
    cfg = setup_cfg(model_name, model_config)
    demo = VisualizationDemo(cfg, args=None)
    demo.predictor.model.model_vision.test_score_thresh = default_score_threshold  # todo 可能之后会设置成更低的，现在设置成一个较低的方便filter
    model2demo[model_name] = demo


def save_object_clip(object_config, frame_config):
    pass


def process_text_prompt(config, question):
    if config['text_prompt'] == 'question':
        text_prompt = question # 这里考虑整个question都做为prompt
    else:
        text_prompt = config['text_prompt']

    return text_prompt

def object_ape(frames, config, question):
    # text prompt = None时进行close-set detect，也就是会detect label set里所有的object
    # 否则只会detect text prompt里的东西
    demo = model2demo[config['model']]
    start_time = time.monotonic()

    res_list = []
    confidence_threshold = config['confidence_threshold']
    # if isinstance(confidence_threshold, float) and confidence_threshold < 1:
    #     demo.predictor.model.model_vision.test_score_thresh = confidence_threshold

    score_list = []

    text_prompt = process_text_prompt(config, question)
    for img_id, img in enumerate(tqdm.tqdm(frames)):
        # use PIL, to be consistent with evaluation
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        predictions, visualized_output, visualized_outputs, metadata = demo.run_on_image(
            img,
            text_prompt=text_prompt,
            with_box=True,
            with_mask=False,
            with_sseg=False,
        )
        scores = predictions["instances"].scores.cpu().numpy()
        score_list.append(scores)

        # plt.imshow(visualized_output.get_image())
        # plt.show()

        res = ""
        with_box = True
        if "instances" in predictions:
            results = instances_to_coco_json(
                predictions["instances"].to(demo.cpu_device), img_id
            )
            if with_box:
                for result_id, result in enumerate(results):
                    res += metadata.thing_classes[result["category_id"]] + ": ["
                    for idx, box in enumerate(result['bbox']):
                        if idx != 3:
                            res += str(int(box)) + ", "
                        else:
                            res += str(int(box))
                    res += "]; "
            else:
                for result_id, result in enumerate(results):
                    res += metadata.thing_classes[result["category_id"]] + ", "

        if len(res) > 0:
            if with_box:
                res_list.append(res[:-2])
            else:
                res_list.append(res)
        else:
            res_list.append("")
    # pickle.dump(res_list, open(res_list_path, 'wb'))

    end_time = time.monotonic()
    processing_time = end_time - start_time

    # 重置confidence threshold
    # demo.predictor.model.model_vision.test_score_thresh = demo.predictor.cfg.model.model_vision.test_score_thresh

    return {
        'object': res_list,
        'score_list': score_list,
        "processing_time": processing_time,
        "config": config,
    }

def get_ape_config4store_path(config):
    config4store_path = config.copy()
    if isinstance(config['text_prompt'], dict):
        config4store_path['text_prompt'] = 'question'
    elif isinstance(config['text_prompt'], str):
        return config4store_path
    else:
        raise TypeError
    return

def filter_result_by_score_threshold(default_result, score_threshold):
    result = default_result.copy()
    new_score_list = []
    new_objects = []
    for id, (scores, objects_str) in enumerate(zip(result["score_list"], result["object"])):
        qualified_ids = [i for i, score in enumerate(scores) if score >= score_threshold]
        new_score_list.append(scores[qualified_ids])
        object_list = objects_str.split('; ')
        new_object_list = [object_list[i] for i in qualified_ids]
        new_object_str = "; ".join(new_object_list)
        new_objects.append(new_object_str)
    result["object"] = new_objects
    result["score_list"] = new_score_list
    return result


force_overwrite = False
def warm_up():
    print('Start warm up with a sample video')
    for model in model2demo.keys():
        print(f'Warm up with model: {model}')
        config = PrepConfig.object_default_config()
        config['model'] = model
        object_ape([np.random.random((200, 200, 3)).astype(np.float32) for _ in range(3)], config, '')
    print('Finish warm up with a sample video')

def main():
    # Set up a socket server
    # server_config = json.load(open('../server_config.json', 'r'))
    port = server_config[modality]['port']
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen()
    warm_up()
    print("object Server is listening...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        data = client_socket.recv(8192)

        # Deserialize datas

        dataset_name, video_name, config, question = pickle.loads(data)
        frame_idx = config.get("frame_idx")
        if dataset_name is None and video_name is None and config is None:
            print(f'{modality} Server closed!')
            break

        store_path = preprocess_store_path(config['modality'], config, video_name, dataset_name)
        exist_result = os.path.exists(store_path)

        default_score_config = config.copy()
        default_score_config[
            'confidence_threshold'] = default_score_threshold  # 先check一下在默认threshold下是否有处理结果 有的话可以直接返回threshold的结果（通常这个threshold是最低的一个值）
        default_result_path = preprocess_store_path(config['modality'], default_score_config, video_name, dataset_name)
        exists_default_result = os.path.exists(default_result_path)
        frame_config = extract_frame_config(config)
        frame_path = preprocess_store_path('frame', frame_config, video_name, dataset_name)
        frame_result = pickle.load(open(frame_path, 'rb'))
        check_config_consistency(frame_result['config'], config)
        frames = frame_result['frames']
        if frame_idx is not None:
            frames = [frames[i] for i in frame_idx]
        if config['text_prompt'] is not None: # 在线动态prompt情况
            if exist_result and not force_overwrite:  # 先检查是否已经处理过，如果处理过且不强制重新处理，直接返回地址
                print(f'Config {config} has been processed before, skip processing!')
                # return store_path
            else:
                result = object_ape(frames, config, question)
                pickle.dump(result, open(store_path, 'wb'))
                del result
        else: # 离线detect所有possible object情况
            if not exists_default_result or force_overwrite:
                assert config['modality'] == modality
                result = object_ape(frames, config, question)
                pickle.dump(result, open(store_path, 'wb'))
                del result
            elif exists_default_result:
                print('Found default result, return the filtering result.')
                result = filter_result_by_score_threshold(pickle.load(open(default_result_path, 'rb')),
                                                          config['confidence_threshold'])
                pickle.dump(result, open(store_path, 'wb'))
                del result
        # Send back result
        client_socket.send(pickle.dumps(store_path))
        client_socket.close()


if __name__ == "__main__":
    main()
