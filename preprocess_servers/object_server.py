import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import itertools
import socket
import pickle
import time
from abc import ABC, abstractmethod
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import RTDETR

from ape_tools.ape_api import setup_cfg
from util.preprocess_constant import ObjModelName, VQADataset
from preprocess_servers.APE.predictor_lazy import VisualizationDemo
from preprocess_servers.tracker.byte_tracker import BYTETracker
import clip
import torchvision.transforms as T
import random as rd

from vidrag_pipeline.scene_graph import generate_scene_graph_description
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

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
import matplotlib.pyplot as plt

from omegaconf import OmegaConf


def check_config_consistency(frame_config, object_config):
    assert frame_config['num_frame'] == object_config['num_frame']
    assert frame_config['sampling_method'] == object_config['sampling_method']


default_confidence_threshold = 0.2


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, dataset_name, video_name, config, question=None):
        pass

    @abstractmethod
    def del_model(self):
        pass


class Track:
    def __init__(self, track_frame_bbox, category):
        self.track_frame_bbox = track_frame_bbox  # [(frame_id, bbox), ...]
        self.category = category


def warm_up_model():
    config = {
        'num_frame': 4,
        # 'fps': 15,
        'confidence_threshold': 0.3,
        # 'model': ObjModelName.RTDETR,
        'model': ObjModelName.APE_L_D,
        'retrieve_threshold': 0.25,
        'text_prompt': None,
    }
    dataset_name = VQADataset.SampleVideo
    video_name = '3269139059.mp4'

    keyword = 'woman in red'
    detect_model.set_detect_threshold(config['confidence_threshold'])
    result = detect_model.detect(dataset_name, video_name, config, show=True, use_track=True)


class APEDetector(ObjectDetector):
    def __init__(self, model_name='APELD'):
        self.model2config = OmegaConf.load(
            os.path.join(os.getenv('HOME'), 'code', 'Video-RAG-master-main', 'ape.yaml')).model2config
        self.model_name = model_name
        model_config = self.model2config[self.model_name]
        cfg = setup_cfg(model_name, model_config)
        demo = VisualizationDemo(cfg, args=None)
        self.detect_threshold = default_confidence_threshold
        self.model = demo
        self.set_detect_threshold(self.detect_threshold)
        # demo.predictor.model.model_vision.test_score_thresh = self.detect_threshold  # todo 可能之后会设置成更低的，现在设置成一个较低的方便filter

    def del_model(self):
        del self.model

    def set_detect_threshold(self, threshold):
        self.detect_threshold = threshold
        self.model.predictor.model.model_vision.test_score_thresh = self.detect_threshold  # todo 可能之后会设置成更低的，现在设置成一个较低的方便filter

    def load_frames(self, config, video_name, dataset_name):
        frame_config = extract_frame_config(config)
        frame_path = preprocess_store_path('frame', frame_config, video_name, dataset_name)
        frame_result = pickle.load(open(frame_path, 'rb'))
        check_config_consistency(frame_result['config'], config)
        frames = frame_result['frames']
        return frames

    def process_object_result(self, video_bboxs, trackid2results=None):
        # 分配track_id, 把属于同一个track的bboxs放在同一个list里，其他的object每个都用一个list包装
        # track_result: [
        # [(frame_id, bbox_id), ....]
        # ]
        frame2trackid = defaultdict(dict)
        # track_id_iter = next(iter(0))
        trackid_counter = itertools.count(0)
        if trackid2results:
            if min(trackid2results.keys()) != 0:
                trackid_offset = min(trackid2results.keys())
                new_trackid2results = {
                    trackid - trackid_offset: trackid2results[trackid] for trackid in trackid2results.keys()
                }
                trackid2results = new_trackid2results
            trackid_counter = itertools.count(len(trackid2results.keys()))

        processed_obj = set()
        tracks = []
        for frame_id, frame_bboxs in enumerate(video_bboxs):
            for bbox_id, frame_bbox in enumerate(frame_bboxs):
                obj_location = (frame_id, bbox_id)
                if obj_location in processed_obj:
                    continue
                # 如果当前obj不在任何一个track里面就当作一个只有一帧的track
                trackid = None
                cur_track_result = [obj_location]
                if trackid2results is not None:
                    found_track = False
                    for trackid, track_result in trackid2results.items():
                        if found_track:
                            break
                        if obj_location in track_result:
                            cur_track_result = track_result
                            found_track = True
                            for track_obj_location in track_result:
                                processed_obj.add(track_obj_location)
                if trackid is None:
                    trackid = next(trackid_counter)
                processed_obj.add(obj_location)
                track_category = frame_bbox[-1]
                tracks.append(
                    Track([[f_id, video_bboxs[f_id][bb_id][0:4]] for (f_id, bb_id) in cur_track_result],
                          track_category))
                frame2trackid[frame_id][trackid] = [track_category, frame_bbox[0:4]]
        return tracks

    def detect(self, dataset_name, video_name, config, question=None, show=False, use_track=False):
        start_time = time.monotonic()
        video_path = get_video_path(dataset_name, video_name)
        tracks, frames = self.object_detect(video_path, config, question=question, show=show, use_track=use_track)
        trackid2clip_emb, trackid2dinov2_emb, trackid2category, trackid2frame, frame2trackid = self.track_embed(tracks,
                                                                                                                frames)
        uid2clip, frame2uid, uid2frame, uid2category = self.reid(frame2trackid, trackid2frame, trackid2category,
                                                                 trackid2clip_emb, trackid2dinov2_emb)
        object_result = process_frame2uid(frame2uid)
        end_time = time.monotonic()
        processing_time = end_time - start_time
        return {
            'object': object_result,
            'uid2clip': uid2clip,
            'uid2category': uid2category,
            'score_list': None,
            "processing_time": processing_time,
            "config": config,
        }

    def object_detect(self, video_path, config, question=None, show=False, use_track=True):
        if config.get('confidence_threshold'):
            self.set_detect_threshold(config['confidence_threshold'])
        args = OmegaConf.load('../bytetracker.yaml')
        # args.track_thresh = self.detect_threshold
        tracker = BYTETracker(args, frame_rate=30)
        # frames = self.load_frames(config, video_name, dataset_name)
        num_frame = config['num_frame']
        frames, frame_idxs = sample_video_frames(video_path, num_frame)
        # text prompt = None时进行close-set detect，也就是会detect label set里所有的object
        # 否则只会detect text prompt里的东西

        res_list = []
        confidence_threshold = config['confidence_threshold']
        # if isinstance(confidence_threshold, float) and confidence_threshold < 1:
        #     demo.predictor.model.model_vision.test_score_thresh = confidence_threshold

        score_list = []

        text_prompt = process_text_prompt(config, question)
        video_bboxs = []
        track_results = None
        for img_id, img in enumerate(tqdm.tqdm(frames)):
            # use PIL, to be consistent with evaluation
            img_size = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            predictions, visualized_output, visualized_outputs, metadata = self.model.run_on_image(
                img,
                text_prompt=text_prompt,
                with_box=True,
                with_mask=False,
                with_sseg=False,
            )
            scores = predictions["instances"].scores.cpu().numpy()
            score_list.append(scores)

            if show:
                plt.imshow(visualized_output.get_image())
                plt.show()

            res = ""
            with_box = True
            frame_bboxs = []
            if "instances" in predictions:
                results = instances_to_coco_json(
                    predictions["instances"].to(self.model.cpu_device), img_id
                )
                for result in results:
                    frame_bbox = []
                    frame_bbox.extend(result['bbox'])
                    frame_bbox.append(result['score'])
                    frame_bbox.append(metadata.thing_classes[result["category_id"]])
                    frame_bboxs.append(frame_bbox)
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

            video_bboxs.append(frame_bboxs)
            if len(res) > 0:
                if with_box:
                    res_list.append(res[:-2])
                else:
                    res_list.append(res)
            else:
                res_list.append("")

            track_input = np.array([frame_bbox[0:6] for frame_bbox in frame_bboxs])
            if use_track:
                frame_track_results = tracker.update(track_input, img_info=img_size, img_size=img_size)
                if track_results is None:
                    track_results = defaultdict(list)
                for frame_track_result in frame_track_results:
                    track_id = frame_track_result.track_id
                    track_results[track_id].append((img_id, frame_track_result.bbox_id))
            print(track_results)
        # pickle.dump(res_list, open(res_list_path, 'wb'))


        return self.process_object_result(video_bboxs, track_results), frames

    def track_embed(self, tracks, frames):
        # tracks = [[(frame_id, bbox, category), .... ], ... , [...]]
        print('generate the CLIP and DINOv2 embedding for each object ID')

        dinov2_transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        trackid2clip_emb = {}
        trackid2dinov2_emb = {}
        trackid2category = {}
        trackid2frame = defaultdict(set)
        frame2trackid = defaultdict(dict)
        for track_id, track in enumerate(tracks):
            track_clip_embeds = []
            track_dinov2_embeds = []
            trackid2category[track_id] = track.category
            for (frame_id, bbox,) in track.track_frame_bbox:
                trackid2frame[track_id].add(frame_id)
                frame2trackid[frame_id][track_id] = [track.category, bbox]
                x, y, w, h = bbox
                # left_top_x = int(x - w / 2)
                # left_top_y = int(y - h / 2)
                # right_bottom_x = int(x + w / 2)
                # right_bottom_y = int(y + h / 2)
                left_top_x = int(x)
                left_top_y = int(y)
                right_bottom_x = int(x + w)
                right_bottom_y = int(y + h)
                frame = frames[frame_id]
                cropped_region = frame[left_top_y:right_bottom_y, left_top_x:right_bottom_x]
                resized_cropped_region = cv2.resize(cropped_region, (224, 224))
                img = resized_cropped_region
                # img = cv2.cvtColor(resized_cropped_region, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                clip_input = clip_transform(img).unsqueeze(0).cuda()
                dinov2_input = dinov2_transform(img).unsqueeze(0).cuda()
                with torch.no_grad():
                    clip_feature = clip_model.encode_image(clip_input).cpu()
                    dinov2_feature = dinov2_model(dinov2_input).cpu()
                track_clip_embeds.append(clip_feature)
                track_dinov2_embeds.append(dinov2_feature)
            clip_emb = torch.cat(track_clip_embeds, dim=0)
            average_clip_emb = torch.mean(clip_emb, dim=0).cpu()
            trackid2clip_emb[track_id] = average_clip_emb

            dinov2_emb = torch.cat(track_dinov2_embeds, dim=0)
            average_dinov2_emb = torch.mean(dinov2_emb, dim=0).cpu()
            trackid2dinov2_emb[track_id] = average_dinov2_emb

        return trackid2clip_emb, trackid2dinov2_emb, trackid2category, trackid2frame, frame2trackid

    def reid(self, frame2trackid, trackid2frame, trackid2category, trackid2clip_emb, trackid2dinov2_emb):
        print('Reid for the track.')

        def clip_similarity_score(obj1, obj2, x0=0.925, slope=20):
            clip_emb1 = trackid2clip_emb[obj1]
            clip_emb2 = trackid2clip_emb[obj2]
            cosine_score = np.dot(clip_emb1, clip_emb2) / (np.linalg.norm(clip_emb1) * np.linalg.norm(clip_emb2))
            clip_score = 1 / (1 + np.exp(-slope * (cosine_score - x0)))
            return clip_score

        def dinov2_similarity_score(obj1, obj2, x0=0.5, slope=4.1):
            dinov2_emb1 = trackid2dinov2_emb[obj1]
            dinov2_emb2 = trackid2dinov2_emb[obj2]
            cosine_score = np.dot(dinov2_emb1, dinov2_emb2) / (
                    np.linalg.norm(dinov2_emb1) * np.linalg.norm(dinov2_emb2))
            # dinov2_score = 1 / (1 + np.exp(-slope * (cosine_score - x0)))
            dinov2_score = cosine_score
            return dinov2_score

        def hard_constraint(obj1, obj2):
            frame1 = set(trackid2frame[obj1])
            frame2 = set(trackid2frame[obj2])
            if len(frame1.intersection(frame2)) > 0:  # if two tracked objects co-exist, they cannot be the same object
                return False
            return True

        def compute_score(obj1, obj2, clip_weight=0.15, dinov2_weight=0.85):
            if not hard_constraint(obj1, obj2):
                return 0
            clip_score = clip_similarity_score(obj1, obj2)
            dinov2_score = dinov2_similarity_score(obj1, obj2)

            return clip_weight * clip_score + dinov2_weight * dinov2_score

        def check_group(tid, uid):
            """tid should has score > 0.5 for all uid objects, and at least one score > 0.62"""
            sgn = False
            for t in uid2tids[uid]:
                if compute_score(tid, t) < 0.5:
                    return False
                if compute_score(tid, t) >= 0.62:
                    sgn = True
            return sgn

        uid2tids = defaultdict(list)
        tid2uid = dict()

        for frame in frame2trackid:
            cur_track_ids = frame2trackid[frame]
            for tid in cur_track_ids:
                if tid in tid2uid:
                    continue
                sgn = False
                for uid in uid2tids:
                    if check_group(tid, uid):
                        uid2tids[uid].append(tid)
                        tid2uid[tid] = uid
                        sgn = True
                        break
                if sgn == False:
                    uid = len(uid2tids)
                    uid2tids[uid].append(tid)
                    tid2uid[tid] = uid

        frame2uid = defaultdict(dict)
        uid2frame = defaultdict(list)
        uid2category = dict()
        uid2clipemb = defaultdict(list)
        uid2clip = dict()
        for frame in frame2trackid:
            for tid in frame2trackid[frame]:
                frame2uid[frame][tid2uid[tid]] = frame2trackid[frame][tid]
        for uid in uid2tids:
            tids = uid2tids[uid]
            for tid in tids:
                uid2frame[uid] += trackid2frame[tid]
                uid2clipemb[uid].append(trackid2clip_emb[tid])

        for uid in uid2clipemb:
            emb = torch.stack(uid2clipemb[uid], dim=0)
            emb = torch.mean(emb, dim=0)
            uid2clip[uid] = emb
        # save_file = os.path.join(video_dir, 'uid2clip.pkl')
        # with open(save_file, 'wb') as f:
        #     pickle.dump(uid2clip, f)

        # reid_file = os.path.join(video_dir, 'reid.pkl')
        for uid in uid2tids:
            uid2category[uid] = trackid2category[uid2tids[uid][0]]
        # with open(reid_file, 'wb') as f:
        #     pickle.dump([frame2uid, uid2frame, uid2category], f)
        return uid2clip, frame2uid, uid2frame, uid2category


clip_model, clip_transform = clip.load(
    os.path.join(os.getenv('HOME'), "model/tool_models/CLIP/ViT-B-32.pt"),
    device="cuda")
dinov2_model = torch.hub.load(
    os.path.join(os.getenv('HOME'), 'model/tool_models/facebookresearch_dinov2_main'),
    'dinov2_vitg14', source="local").cuda()


def process_frame2uid(frame2uid, bbox_mode='left_top'):
    object_result = []
    for frame_uid in frame2uid.values():
        str_result = ''
        for uid, object in frame_uid.items():
            category_name, bbox = object[0], object[1]
            bbox = [int(i) for i in bbox]
            if bbox_mode == 'center':
                bbox[0] = int(bbox[0] - bbox[2] / 2)
                bbox[1] = int(bbox[1] - bbox[3] / 2)
            category_name = f'{category_name}_{uid}'
            str_result += f'{category_name}: {bbox}; '
        if len(str_result) > 0:
            str_result = str_result[:-2]
        object_result.append(str_result)

    return object_result


class UltralyticsDetector(ObjectDetector):

    def __init__(self, model_name='RTDETR'):
        self.detect_threshold = default_confidence_threshold
        if model_name == 'RTDETR':
            self.tracking_model = RTDETR(os.path.join(os.getenv('HOME'), 'model/tool_models/tracking/rtdetr-l.pt'))
        else:
            raise NotImplementedError
        self.model_name = model_name

    def del_model(self):
        del self.tracking_model

    def warm_up(self):
        pass

    def detect(self, dataset_name, video_name, config, question=None, show=False, use_track=True):
        if config.get('confidence_threshold'):
            self.set_detect_threshold(config['confidence_threshold'])
        start_time = time.monotonic()
        video_path = get_video_path(dataset_name, video_name)
        frame2trackid, trackid2frame, trackid2category = self.track(video_path, config, show=show)
        track_id2avg_clip_emb, track_id2avg_dinov2_emb = self.track_embed(video_path, config, frame2trackid,
                                                                          trackid2frame)

        uid2clip, frame2uid, uid2frame, uid2category = self.reid(frame2trackid, trackid2frame, trackid2category,
                                                                 track_id2avg_clip_emb, track_id2avg_clip_emb)

        object_result = process_frame2uid(frame2uid, bbox_mode='center')
        end_time = time.monotonic()
        processing_time = end_time - start_time
        return {
            'object': object_result,
            'uid2clip': uid2clip,
            'score_list': None,
            "processing_time": processing_time,
            "config": config,
            "uid2category": uid2category
        }

    def track(self, video_path, config, show=False):
        print('Begin tracking')
        cap = cv2.VideoCapture(video_path)
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if config.get('fps'):
            sample_fps = config['fps']
            assert sample_fps <= video_fps
            num_frame = int(total_frame_count / video_fps * sample_fps)
            config['num_frame'] = num_frame
        else:
            num_frame = config['num_frame']
        confidence_threshold = config['confidence_threshold']
        frame2trackid = defaultdict(dict)
        trackid2frame = defaultdict(list)
        trackid2category_cnt = defaultdict(list)
        # Loop through the video frames
        sparse_frames, frame_idxs = sample_video_frames(video_path, num_frame)
        for frame, frame_idx in zip(sparse_frames, frame_idxs):
            results = self.tracking_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False,
                                                conf=confidence_threshold)
            boxes = results[0].boxes
            if boxes.id is None:
                continue
            cls = boxes.cls.numpy()
            id = boxes.id.numpy()
            xywh = boxes.xywh.numpy()  # notice: x, y are the center coordinates
            num_boxes = id.shape[0]
            for i in range(num_boxes):
                track_id = int(id[i])
                category = id2category[cls[i]]
                box = list(xywh[i])
                frame2trackid[frame_idx][track_id] = [category, box]
                trackid2frame[track_id].append(frame_idx)
                trackid2category_cnt[track_id].append(category)
            if show:
                annotated_frame = results[0].plot()
                # cv2.imshow("RTDETR Tracking", annotated_frame)
                plt.imshow(annotated_frame)
                plt.show()
        cv2.destroyAllWindows()
        trackid2category = dict()
        for track_id in trackid2category_cnt:
            category_cnt = trackid2category_cnt[track_id]
            most_common_category = max(set(category_cnt), key=category_cnt.count)
            trackid2category[track_id] = most_common_category

        return frame2trackid, trackid2frame, trackid2category

    def track_embed(self, video_path, config, frame2trackid, trackid2frame, sample_num=5):
        print('generate the CLIP and DINOv2 embedding for each tracking ID')
        num_frame = config['num_frame']

        dinov2_transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        cap = cv2.VideoCapture(video_path)
        frame2select = defaultdict(list)
        track_id2clip_emb = defaultdict(list)
        track_id2dinov2_emb = defaultdict(list)
        for track_id in trackid2frame:
            frame_ids = trackid2frame[track_id]
            rd.setstate(rd_state)
            selected_frame_ids = rd.sample(frame_ids, min(len(frame_ids), sample_num))
            for frame in selected_frame_ids:
                frame2select[frame].append(track_id)

        sparse_frames, frame_idxs = sample_video_frames(video_path, num_frame)
        for frame, frame_idx in zip(sparse_frames, frame_idxs):
            for track_id in frame2select[frame_idx]:
                bbox = frame2trackid[frame_idx][track_id][1]
                x, y, w, h = bbox
                left_top_x = int(x - w / 2)
                left_top_y = int(y - h / 2)
                right_bottom_x = int(x + w / 2)
                right_bottom_y = int(y + h / 2)
                cropped_region = frame[left_top_y:right_bottom_y, left_top_x:right_bottom_x]
                resized_cropped_region = cv2.resize(cropped_region, (224, 224))
                img = resized_cropped_region
                # img = cv2.cvtColor(resized_cropped_region, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                clip_input = clip_transform(img).unsqueeze(0).cuda()
                dinov2_input = dinov2_transform(img).unsqueeze(0).cuda()
                with torch.no_grad():
                    clip_feature = clip_model.encode_image(clip_input).cpu()
                    dinov2_feature = dinov2_model(dinov2_input).cpu()
                track_id2clip_emb[track_id].append(clip_feature)
                track_id2dinov2_emb[track_id].append(dinov2_feature)
        cap.release()
        track_id2avg_clip_emb = dict()
        track_id2avg_dinov2_emb = dict()
        for track_id in track_id2clip_emb:
            clip_emb = torch.cat(track_id2clip_emb[track_id], dim=0)
            average_clip_emb = torch.mean(clip_emb, dim=0).cpu()
            track_id2avg_clip_emb[track_id] = average_clip_emb
        for track_id in track_id2dinov2_emb:
            dinov2_emb = torch.cat(track_id2dinov2_emb[track_id], dim=0)
            average_dinov2_emb = torch.mean(dinov2_emb, dim=0).cpu()
            track_id2avg_dinov2_emb[track_id] = average_dinov2_emb

        return track_id2avg_clip_emb, track_id2avg_dinov2_emb

    def reid(self, frame2trackid, trackid2frame, trackid2category, trackid2clip_emb, trackid2dinov2_emb):
        print('Reid for the track.')

        def clip_similarity_score(obj1, obj2, x0=0.925, slope=20):
            clip_emb1 = trackid2clip_emb[obj1]
            clip_emb2 = trackid2clip_emb[obj2]
            cosine_score = np.dot(clip_emb1, clip_emb2) / (np.linalg.norm(clip_emb1) * np.linalg.norm(clip_emb2))
            clip_score = 1 / (1 + np.exp(-slope * (cosine_score - x0)))
            return clip_score

        def dinov2_similarity_score(obj1, obj2, x0=0.5, slope=4.1):
            dinov2_emb1 = trackid2dinov2_emb[obj1]
            dinov2_emb2 = trackid2dinov2_emb[obj2]
            cosine_score = np.dot(dinov2_emb1, dinov2_emb2) / (
                    np.linalg.norm(dinov2_emb1) * np.linalg.norm(dinov2_emb2))
            # dinov2_score = 1 / (1 + np.exp(-slope * (cosine_score - x0)))
            dinov2_score = cosine_score
            return dinov2_score

        def hard_constraint(obj1, obj2):
            frame1 = set(trackid2frame[obj1])
            frame2 = set(trackid2frame[obj2])
            if len(frame1.intersection(frame2)) > 0:  # if two tracked objects co-exist, they cannot be the same object
                return False
            return True

        def compute_score(obj1, obj2):
            if not hard_constraint(obj1, obj2):
                return 0
            clip_score = clip_similarity_score(obj1, obj2)
            dinov2_score = dinov2_similarity_score(obj1, obj2)
            return 0.15 * clip_score + 0.85 * dinov2_score

        def check_group(tid, uid):
            """tid should has score > 0.5 for all uid objects, and at least one score > 0.62"""
            sgn = False
            for t in uid2tids[uid]:
                if compute_score(tid, t) < 0.5:
                    return False
                if compute_score(tid, t) >= 0.62:
                    sgn = True
            return sgn

        uid2tids = defaultdict(list)
        tid2uid = dict()

        for frame in frame2trackid:
            cur_track_ids = frame2trackid[frame]
            for tid in cur_track_ids:
                if tid in tid2uid:
                    continue
                sgn = False
                for uid in uid2tids:
                    if check_group(tid, uid):
                        uid2tids[uid].append(tid)
                        tid2uid[tid] = uid
                        sgn = True
                        break
                if sgn == False:
                    uid = len(uid2tids)
                    uid2tids[uid].append(tid)
                    tid2uid[tid] = uid

        frame2uid = defaultdict(dict)
        uid2frame = defaultdict(list)
        uid2category = dict()
        uid2clipemb = defaultdict(list)
        uid2clip = dict()
        for frame in frame2trackid:
            for tid in frame2trackid[frame]:
                frame2uid[frame][tid2uid[tid]] = frame2trackid[frame][tid]
        for uid in uid2tids:
            tids = uid2tids[uid]
            for tid in tids:
                uid2frame[uid] += trackid2frame[tid]
                uid2clipemb[uid].append(trackid2clip_emb[tid])

        for uid in uid2clipemb:
            emb = torch.stack(uid2clipemb[uid], dim=0)
            emb = torch.mean(emb, dim=0)
            uid2clip[uid] = emb
        # save_file = os.path.join(video_dir, 'uid2clip.pkl')
        # with open(save_file, 'wb') as f:
        #     pickle.dump(uid2clip, f)

        # reid_file = os.path.join(video_dir, 'reid.pkl')
        for uid in uid2tids:
            uid2category[uid] = trackid2category[uid2tids[uid][0]]
        # with open(reid_file, 'wb') as f:
        #     pickle.dump([frame2uid, uid2frame, uid2category], f)
        return uid2clip, frame2uid, uid2frame, uid2category

    def set_detect_threshold(self, threshold):
        self.detect_threshold = threshold
        pass


def process_text_prompt(config, question):
    if not config.get('question'):
        return None
    if config['text_prompt'] == 'question':
        text_prompt = question  # 这里考虑整个question都做为prompt
    else:
        text_prompt = config['text_prompt']

    return text_prompt


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

detect_model = None


def load_detect_model(model_name):
    global detect_model
    if detect_model:
        if detect_model.model_name == model_name:
            return detect_model
        else:
            detect_model.del_model()

    if model_name in ObjModelName.ape_series_model_names:
        detect_model = APEDetector(model_name=model_name)
        warm_up_model()

    elif model_name in ObjModelName.ultralytics_series_model_names:
        detect_model = UltralyticsDetector(model_name=model_name)
        warm_up_model()

    return detect_model


def main():
    # Set up a socket server
    # server_config = json.load(open('../server_config.json', 'r'))
    port = server_config[modality]['port']
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen()

    print("object Server is listening...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        data = client_socket.recv(8192)

        # Deserialize datas

        dataset_name, video_name, config, question = pickle.loads(data)


        # frame_idx = config.get("frame_idx")
        if dataset_name is None and video_name is None and config is None and question is None:
            print(f'{modality} Server closed!')
            break

        store_path = preprocess_store_path(config['modality'], config, video_name, dataset_name)
        exist_result = os.path.exists(store_path)

        # default_score_config = config.copy()
        # default_score_config[
        #     'confidence_threshold'] = default_confidence_threshold  # 先check一下在默认threshold下是否有处理结果 有的话可以直接返回threshold的结果（通常这个threshold是最低的一个值）
        # default_result_path = preprocess_store_path(config['modality'], default_score_config, video_name, dataset_name)
        # exists_default_result = os.path.exists(default_result_path)
        # frame_config = extract_frame_config(config)
        # frame_path = preprocess_store_path('frame', frame_config, video_name, dataset_name)
        # frame_result = pickle.load(open(frame_path, 'rb'))
        # check_config_consistency(frame_result['config'], config)
        # frames = frame_result['frames']

        # if frame_idx is not None:
        #     frames = [frames[i] for i in frame_idx]

        # if config.get('text_prompt') and config['text_prompt'] is not None:  # 在线动态prompt情况
        #     if exist_result and not force_overwrite:  # 先检查是否已经处理过，如果处理过且不强制重新处理，直接返回地址
        #         print(f'Config {config} has been processed before, skip processing!')
        #         # return store_path
        #     else:
        #         result = detect_model.detect(dataset_name, video_name, config, question)
        #         pickle.dump(result, open(store_path, 'wb'))
        #         del result
        # else:  # 离线detect所有possible object情况
        if force_overwrite or not exist_result:
            model_name = config['model']
            load_detect_model(model_name)
            assert config['modality'] == modality
            result = detect_model.detect(dataset_name, video_name, config, show=True, use_track=True)
            pickle.dump(result, open(store_path, 'wb'))
            del result
        else:
            print(f'Config {config} has been processed before, skip processing!')

        # Send back result
        client_socket.send(pickle.dumps(store_path))
        client_socket.close()


if __name__ == '__main__':
    main()
# %%

# from sklearn.metrics.pairwise import cosine_similarity

# sentence_model, _ = clip.load("ViT-B/32", device='cuda')

# def encode_sentences(sentence_list, model_name, sentence_model):
#     '''given a list of sentences, return the embeddings for them using the sentence encoder model'''
#     assert model_name == 'clip'
#     emb_list = []
#     device = "cuda"
#     num_iter = 1
#     with torch.no_grad():
#         for sentence in sentence_list:  # todo 循环三次进行measure
#             emb_list.append(sentence_model.encode_text(clip.tokenize([sentence]).to(device)).cpu().numpy())
#     emb_list = np.concatenate(emb_list, axis=0)
#     return emb_list
#
#
#
#
#
# def retrieve_candidate_objects(description, uid2emb,uid2category, retrieve_threshold=0.26):
#     def select_elements_with_repetition(lst, count):
#         if not lst:  # 如果列表为空，返回空列表
#             return []
#
#         result = []
#         for i in range(count):
#             # 使用模运算来循环获取元素
#             result.append(lst[i % len(lst)])
#         return result
#
#     def compute_cosine_similarity(target_embedding, embedding_list):
#         embedding_list_np = [embedding.numpy() for embedding in embedding_list]
#         target_embedding_tensor = target_embedding.reshape(1, -1)
#         # Compute cosine similarity
#         start = time.monotonic()
#         similarity_scores = cosine_similarity(target_embedding_tensor, embedding_list_np)
#         end = time.monotonic()
#         return similarity_scores.reshape(-1)
#
#     des_emb = encode_sentences([f"a photo of a {description}."], model_name='clip', sentence_model=sentence_model)
#     scores = compute_cosine_similarity(des_emb, list(uid2emb.values()))
#     indices = np.where(scores >= retrieve_threshold)[0]
#     selected_categories = [uid2category[i] for i in indices]
#     candidate_uids = []
#     for i in indices:
#         candidate_uids.append(list(uid2emb)[i])
#     return candidate_uids, scores
#
#     # main()
#
# def get_retrieved_object_result(object_result, keyword, uid2clip,uid2category, config):
#     # retrieve_threshold = 0.26
#     retrieve_threshold = 0.25
#     if config.get('retrieve_threshold'):
#         retrieve_threshold = config['retrieve_threshold']
#     candidate_uids, retrieve_scores = retrieve_candidate_objects(keyword, uid2clip,uid2category, retrieve_threshold=retrieve_threshold)
#     retrieved_object_result = []
#     for frame_result_str in object_result:
#         frame_result = frame_result_str.strip().split(';')
#         retrieved_frame_result = ''
#         for object_str in frame_result:
#             tmp = object_str.split(':')
#             org_category_name, bbox = tmp[0], tmp[1]
#             uid = int(org_category_name.split('_')[-1])
#             if uid in candidate_uids:
#                 category_name = keyword
#                 retrieved_frame_result += f'{category_name} / {org_category_name}: {bbox}; '
#         if len(retrieved_frame_result) > 0:
#             retrieved_frame_result = retrieved_frame_result[:-2]
#         if len(retrieved_frame_result) > 0:
#             retrieved_object_result.append(retrieved_frame_result)
#     return retrieved_object_result

# dataset_name = VQADataset.SampleVideo
# video_name = '3269139059.mp4'
#
# config = {
#     'num_frame': 4,
#     # 'fps': 15,
#     'confidence_threshold': 0.3,
#     # 'model': ObjModelName.RTDETR,
#     'model': ObjModelName.APE_L_D,
#     'retrieve_threshold': 0.25,
#     'text_prompt': None,
#     'sampling_method': 'uniform'
# }
# load_detect_model(config['model'])
#
# keyword = 'woman in red'
# detect_model.set_detect_threshold(config['confidence_threshold'])
# result = detect_model.detect(dataset_name, video_name, config, show=True, use_track = True)
# object_result = result['object']
# uid2clip = result['uid2clip']
# uid2category = result['uid2category']


# retrieved_object_result = get_retrieved_object_result(object_result, keyword, uid2clip,uid2category, config,clip.tokenize, sentence_model)
#
#
# def det_preprocess(det_docs, location, relation, number, keyword_extractor):
#     scene_descriptions = []
#
#     for det_doc_per_frame in det_docs:
#         objects = []
#         scene_description = ""
#         if len(det_doc_per_frame) > 0:
#             for obj_id, objs in enumerate(det_doc_per_frame.split(";")):
#                 if isinstance(keyword_extractor, BareKeywordExtractor):  # 针对bare keyword retriever特殊处理
#                     obj_name = f'Object {obj_id}'
#                 else:
#                     obj_name = objs.split(":")[0].strip()
#                 obj_bbox = objs.split(":")[-1].strip()
#                 obj_bbox = ast.literal_eval(obj_bbox)
#                 objects.append({"id": obj_id, "label": obj_name, "bbox": obj_bbox})
#
#             scene_description = generate_scene_graph_description(objects, location, relation, number)
#         scene_descriptions.append(scene_description)
#
#     return scene_descriptions
#
#
# print('scene_description', det_preprocess(retrieved_object_result, True, True, True, None))
