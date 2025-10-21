import itertools
import os
import json
from collections import defaultdict

import pandas as pd
from omegaconf import OmegaConf

from preprocess_constant import *
from pathlib import Path

options_letter = ['A', 'B', 'C', 'D', 'E', 'F']


class VideoDatasetLoader:
    def __init__(self):
        self.dataset2load_func = {
            VQADataset.MSVD_QA: self.load_msvdqa,
            VQADataset.MSRVTT_QA: self.load_msvrtt,
            VQADataset.GroundQA: self.load_goundqa,
            VQADataset.AGQA: self.load_agqa,
            VQADataset.CinePipe: self.load_cinepipe,
            VQADataset.ALLVB: self.load_allvb,
            VQADataset.NextQA: self.load_nextqa,
            VQADataset.SampleVideo: self.load_samplevideo,
        }
        self.dataset2prep_config_set = {
            VQADataset.MSVD_QA: self.msvdqa_prepconfig,
            VQADataset.MSRVTT_QA: self.msvdqa_prepconfig,
            VQADataset.GroundQA: self.groundqa_prepconfig,
            VQADataset.AGQA: self.agqa_prepconfig,
            VQADataset.CinePipe: self.cinepipe_prepconfig,
            VQADataset.ALLVB: self.allvb_prepconfig,
            VQADataset.NextQA: self.nextqa_prepconfig,
            VQADataset.SampleVideo: self.nextqa_prepconfig,
        }
        self.video_formats = ['mp4', 'avi']
        self.option_keys = {
            VQADataset.ALLVB: ['optionA', 'optionB', 'optionC', 'optionD', 'optionE'],
            VQADataset.NextQA: ['a0', 'a1', 'a2', 'a3', 'a4'],
            VQADataset.SampleVideo: ['a0', 'a1', 'a2', 'a3', 'a4']
        }

    def load_dataset(self, dataset_name, top_k_video=None, num_min_query=None,base_video_num=0):
        assert dataset_name in self.dataset2load_func.keys(), 'Dataset {} not supported.'.format(dataset_name)
        return self.dataset2load_func[dataset_name](top_k_video=top_k_video, num_min_query=num_min_query,
                                                    base_video_num=base_video_num)

    def get_video_id2name(self, video_dir, dataset_name):
        # 获取video id对应的video文件名（带后缀）
        video_names = [file for file in os.listdir(video_dir) if
                       any([file.endswith(video_format) for video_format in self.video_formats])]
        video_id2name = {}
        for video_name in video_names:
            if dataset_name == VQADataset.MSRVTT_QA:  # video**.mp4
                video_id = video_name.split('.')[0][5:]
            else:
                video_id = video_name.split('.')[0]
            video_id2name[video_id] = video_name

        return video_id2name

    def process_MC_question(self, dataset_name, question_dict):
        org_question = question_dict['question']
        processed_question = org_question
        for i, option_key in enumerate(self.option_keys[dataset_name]):
            if option_key in question_dict.keys():
                processed_question = processed_question + '\n' + options_letter[i] + ': ' + question_dict[option_key]

        question_dict['question'] = processed_question
        question_dict['org_question'] = org_question
        return question_dict

    def load_msvdqa(self, top_k_video, num_min_query, base_video_num=0):
        dataset_path = os.path.join(os.environ['HOME'], 'dataset', 'MSVD-QA')
        video_dir = os.path.join(dataset_path, 'videos')
        question_files = ['train.jsonl', 'val.jsonl', 'test.jsonl']
        video_id2name = self.get_video_id2name(video_dir, dataset_name=VQADataset.MSVD_QA)
        video_id2question_dicts = defaultdict(list)
        video_id2question_id_cnt = defaultdict(int)
        for question_file in question_files:
            path = os.path.join(dataset_path, question_file)
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()  # 去除首尾空白字符
                    if line:
                        question_dict = json.loads(line)
                        video_id = question_dict['video_id']
                        question_id = video_id2question_id_cnt[video_id]
                        video_id2question_id_cnt[video_id] += 1
                        question_dict['question_id'] = f'{video_id}_{str(question_id)}'
                        video_id2question_dicts[video_id].append(question_dict)
        if top_k_video:
            video_id2question_dicts = self.get_top_k_video(video_id2question_dicts, top_k_video,
                                                           base_video_num=base_video_num)
        if num_min_query:
            video_id2question_dicts = self.filter_video_with_query_num(num_min_query=num_min_query,
                                                                       video_id2questions=video_id2question_dicts)
        return video_id2question_dicts, dataset_path, QuestionType.OE, video_id2name

    def load_msvrtt(self, top_k_video, num_min_query, base_video_num=0):
        dataset_path = os.path.join(os.environ['HOME'], 'dataset', VQADataset.MSRVTT_QA)
        video_dir = os.path.join(dataset_path, 'videos')
        question_files = ['train_qa.json', 'val_qa.json', 'test_qa.json']
        video_id2name = self.get_video_id2name(video_dir, dataset_name=VQADataset.MSRVTT_QA)
        video_id2question_dicts = defaultdict(list)
        video_id2question_id_cnt = defaultdict(int)
        for question_file in question_files:
            path = os.path.join(dataset_path, question_file)
            data = json.load(open(path, 'r'))
            for question_dict in data:
                video_id = str(question_dict['video_id'])
                question_id = video_id2question_id_cnt[video_id]
                video_id2question_id_cnt[video_id] += 1
                question_dict['question_id'] = f'{video_id}_{str(question_id)}'
                video_id2question_dicts[video_id].append(question_dict)
        if top_k_video:
            video_id2question_dicts = self.get_top_k_video(video_id2question_dicts, top_k_video,
                                                           base_video_num=base_video_num)
        if num_min_query:
            video_id2question_dicts = self.filter_video_with_query_num(num_min_query=num_min_query,
                                                                       video_id2questions=video_id2question_dicts)
        return video_id2question_dicts, dataset_path, QuestionType.OE, video_id2name

    def load_goundqa(self, top_k_video, num_min_query, reverse = True, base_video_num=0):
        dataset_path = os.path.join(os.environ['HOME'], 'dataset', VQADataset.GroundQA)
        video_dir = os.path.join(dataset_path, 'videos')
        question_files = ['annotations.EgoTimeQA.json']
        video_id2name = self.get_video_id2name(video_dir, dataset_name=VQADataset.GroundQA)
        video_id2question_dicts = defaultdict(list)
        video_id2question_id_cnt = defaultdict(int)
        if num_min_query is None:
            num_min_query = 100
        for question_file in question_files:
            path = os.path.join(dataset_path, question_file)
            data = json.load(open(path, 'r'))
            for question_dict in data:
                video_id = question_dict['video_id']
                if video_id in video_id2name:
                    question_id = video_id2question_id_cnt[video_id]
                    video_id2question_id_cnt[video_id] += 1
                    question_dict['question_id'] = f'{video_id}_{str(question_id)}'
                    video_id2question_dicts[video_id].append(question_dict)
        if num_min_query is not None:
            video_id2question_dicts = self.filter_video_with_query_num(num_min_query=num_min_query,
                                                                       video_id2questions=video_id2question_dicts)
        if top_k_video:
            video_id2question_dicts = self.get_top_k_video(video_id2question_dicts, top_k_video,
                                                           base_video_num=base_video_num, reverse = reverse)
        # if num_min_query:
        #     video_id2question_dicts = self.filter_video_with_query_num(num_min_query=num_min_query,num_max_query = num_max_query,
        #                                                                video_id2questions=video_id2question_dicts)
        return video_id2question_dicts, dataset_path, QuestionType.OE, video_id2name

    def load_nextqa(self, top_k_video, num_min_query):
        dataset_path = os.path.join(os.environ['HOME'], 'dataset', 'Next-QA')
        video_dir = os.path.join(dataset_path, 'videos')
        question_files = ['train.csv']
        video_id2name = self.get_video_id2name(video_dir)
        video_id2question_dicts = defaultdict(list)
        video_id2question_id_cnt = defaultdict(int)
        for question_file in question_files:
            df = pd.read_csv(os.path.join(dataset_path, question_file))
            for index, row in df.iterrows():
                video_id = str(row['video'])
                if video_id in video_id2name.keys():
                    question_id = video_id2question_id_cnt[video_id]
                    video_id2question_id_cnt[video_id] += 1
                    question_dict = self.process_MC_question(VQADataset.NextQA, row.to_dict().copy())
                    question_dict['question_id'] = f'{video_id}_{str(question_id)}'
                    video_id2question_dicts[video_id].append(question_dict)
        if top_k_video:
            video_id2question_dicts = self.get_top_k_video(video_id2question_dicts, top_k_video)
        if num_min_query:
            video_id2question_dicts = self.filter_video_with_query_num(num_min_query=num_min_query,
                                                                       video_id2questions=video_id2question_dicts)
        return video_id2question_dicts, dataset_path, QuestionType.MC, video_id2name

    def load_samplevideo(self, top_k_video, num_min_query,
                                                    base_video_num):
        dataset_path = os.path.join(os.environ['HOME'], 'dataset', 'sample_video')
        video_dir = os.path.join(dataset_path, 'videos')
        question_files = ['train.csv']
        video_id2name = self.get_video_id2name(video_dir, VQADataset.SampleVideo)
        video_id2question_dicts = defaultdict(list)
        video_id2question_id_cnt = defaultdict(int)
        for question_file in question_files:
            df = pd.read_csv(os.path.join(dataset_path, question_file))
            for index, row in df.iterrows():
                video_id = str(row['video'])
                if video_id in video_id2name.keys():
                    question_id = video_id2question_id_cnt[video_id]
                    video_id2question_id_cnt[video_id] += 1
                    question_dict = self.process_MC_question(VQADataset.NextQA, row.to_dict().copy())
                    question_dict['question_id'] = f'{video_id}_{str(question_id)}'
                    video_id2question_dicts[video_id].append(question_dict)
        return video_id2question_dicts, dataset_path, QuestionType.MC, video_id2name

    def get_dataset_prep_config_set(self, dataset_name, type=PrepConfigSetType.VisiontokenOnly):
        return self.dataset2prep_config_set[dataset_name](type)

    def get_dataset_inference_config_set(self, dataset_name, type=PrepConfigSetType.VisiontokenOnly):
        # 部分config在runtime的时候才用的到，比如retrieve_threshold,因此这里会加入一些额外的config
        return self.dataset2prep_config_set[dataset_name](type, mode='inference')

    def load_cinepipe(self, top_k_video, num_min_query):
        raise NotImplementedError

    def load_agqa(self, top_k_video, num_min_query):
        raise NotImplementedError

    def load_allvb(self, top_k_video, num_min_query):
        dataset_path = os.path.join(os.environ['HOME'], 'dataset', VQADataset.ALLVB)
        video_dir = os.path.join(dataset_path, 'videos')
        question_files = [file for file in os.listdir(video_dir) if file.endswith('.txt')]
        video_id2name = self.get_video_id2name(video_dir)
        video_id2questions = defaultdict(list)
        for question_file in question_files:
            video_id = question_file.split('.')[0]
            question_id = question_file.split('.')[0]
            path = os.path.join(dataset_path, 'videos', question_file)
            with open(path, 'r') as f:
                question = {}
                for line in f:
                    line = line.strip()  # 去除首尾空白字符
                    if line:
                        if line.startswith('Question'):
                            question_class = line.strip().split(': ')[1]
                            question['question_class'] = question_class
                            pass
                        elif line.startswith('-'):
                            option_id, option = line[2], line.strip()[5:]
                            assert option_id in ['A', 'B', 'C', 'D', 'E']
                            question[f'option{option_id}'] = option
                        elif line.startswith('Answer'):
                            answer = line.split(': ')[1]
                            assert len(answer) == 1
                            question['answer'] = answer
                            video_id2questions[video_id].append(
                                self.process_MC_question(VQADataset.ALLVB, question.copy()))
                            question = {}
                        else:  # 问题本身
                            question['question'] = line
        if top_k_video:
            video_id2questions = self.get_top_k_video(video_id2questions, top_k_video)
        if num_min_query:
            video_id2questions = self.filter_video_with_query_num(num_min_query=num_min_query,
                                                                  video_id2questions=video_id2questions)
        return video_id2questions, dataset_path, QuestionType.MC, video_id2name

    def get_top_k_video(self, video2questions, top_k_video, base_video_num=0, reverse = False):
        if top_k_video is None:
            return video2questions
        if len(video2questions) <= top_k_video:
            return video2questions
        video_ids = list(video2questions.keys())
        sorted_video_ids = sorted(video_ids, key=lambda x: len(video2questions[x]), reverse= not reverse)
        return {
            video_id: video2questions[video_id] for video_id in sorted_video_ids[base_video_num:top_k_video]
        }

    def filter_video_with_query_num(self, video_id2questions, num_min_query, num_max_query = None):
        filtered_video_id2questions = defaultdict(list)
        for key, val in video_id2questions.items():
            if num_min_query and len(val) >= num_min_query:
                if num_max_query is not None and len(val) >= num_max_query:
                    continue
                filtered_video_id2questions[key] = val
        return filtered_video_id2questions

    def nextqa_prepconfig(self, type, mode='preprocess'):
        if type == PrepConfigSetType.Combination:
            frame_config_set = OmegaConf.create({
                'modality': [Modality.Frame],
                'num_frame': [16],
                'sampling_method': ['uniform']
            })
            visiontoken_configs_set = OmegaConf.create({
                'modality': [Modality.VisionToken],
            })
            visiontoken_configs_set = OmegaConf.merge(frame_config_set, visiontoken_configs_set)
            ocr_configs_set = OmegaConf.create({
                'modality': [Modality.OCR],
            })
            ocr_configs_set = OmegaConf.merge(frame_config_set, ocr_configs_set)
            asr_configs_set = OmegaConf.create({
                'modality': [Modality.ASR],
                'model': ['whisper-large'],
                'chunk_length': [3]
            })
            caption_configs_set = OmegaConf.create({
                'modality': [Modality.Caption],
                'seconds_per_caption': [2],
                'frames_per_caption': [4],
                'model': [CaptionModelName.lavila]
            })
            object_configs_set = OmegaConf.create({
                'modality': [Modality.Object],
                'num_frame': [16],
                'confidence_threshold': [0.3],
                'model': [ObjModelName.APE_L_D],
            })
            if mode != 'preprocess':
                object_configs_set['retrieve_threshold'] = [0.26]
            clip_configs_set = OmegaConf.create({
                'modality': [Modality.Clip],
            })
            clip_configs_set = OmegaConf.merge(frame_config_set, clip_configs_set)

            modality2config_set = {
                Modality.Frame: frame_config_set,
                Modality.OCR: ocr_configs_set,
                Modality.ASR: asr_configs_set,
                Modality.VisionToken: visiontoken_configs_set,
                Modality.Caption: caption_configs_set,
                Modality.Clip: clip_configs_set,
                Modality.Object: object_configs_set,
            }
        elif type == PrepConfigSetType.CaptionOnly:
            caption_configs_set = OmegaConf.create({
                'modality': [Modality.Caption],
                'seconds_per_caption': [1, 2, 4],
                'frames_per_caption': [1, 2, 4],
                'model': CaptionModelName.all_models
            })

            modality2config_set = {
                Modality.Caption: caption_configs_set,
            }
        elif type == PrepConfigSetType.VisiontokenOnly:
            frame_config_set = OmegaConf.create({
                'modality': [Modality.Frame],
                'num_frame': [4, 8, 16, 32, 64],
                'sampling_method': ['uniform', 'keyframe'],
            })
            visiontoken_configs_set = OmegaConf.create({
                'modality': [Modality.VisionToken],
            })
            visiontoken_configs_set = OmegaConf.merge(frame_config_set, visiontoken_configs_set)
            modality2config_set = {
                Modality.Frame: frame_config_set,
                Modality.VisionToken: visiontoken_configs_set,
            }
        elif type == PrepConfigSetType.ObjectOnly:
            object_config = {
                'modality': [Modality.Object],
                'num_frame': [4, 8, 16, 32],
                'confidence_threshold': [0.3, 0.4, 0.5],
                'model': ObjModelName.all_models,
                # 'retrieve_threshold': [0.26],
                # 'text_prompt': [None],
            }
            modality2config_set = {
                Modality.Object: object_config,
            }
        elif type == PrepConfigSetType.All:
            frame_config_set = OmegaConf.create({
                'modality': [Modality.Frame],
                'num_frame': [4, 8, 16, 32, 64],
                'sampling_method': ['uniform', 'keyframe'],
            })
            visiontoken_configs_set = OmegaConf.create(
                {
                    'modality': [Modality.VisionToken],
                    'num_frame': [4, 8, 16, 32, 64],
                    'sampling_method': ['uniform', 'keyframe'],
                }
            )
            ocr_configs_set = OmegaConf.create({
                'modality': [Modality.OCR],
            })
            ocr_configs_set = OmegaConf.merge(frame_config_set, ocr_configs_set)
            asr_configs_set = OmegaConf.create({
                'modality': [Modality.ASR],
                'model': ASRModelName.all_models,
                'chunk_length': [3, 6, 9]
            })
            caption_configs_set = OmegaConf.create({
                'modality': [Modality.Caption],
                'seconds_per_caption': [1, 2, 4],
                'frames_per_caption': [1, 2, 4],
                'model': CaptionModelName.all_models
            })
            object_configs_set = OmegaConf.create({
                'modality': [Modality.Object],
                'num_frame': [4, 8, 16, 32],
                'confidence_threshold': [0.3, 0.4, 0.5],
                'model': ObjModelName.all_models,
            })

            modality2config_set = {
                Modality.Frame: frame_config_set,
                Modality.OCR: ocr_configs_set,
                Modality.ASR: asr_configs_set,
                Modality.VisionToken: visiontoken_configs_set,
                Modality.Caption: caption_configs_set,
                Modality.Object: object_configs_set,
            }
        else:
            raise NotImplementedError

        return modality2config_set

    def msvdqa_prepconfig(self, type, mode='preprocess'):
        if type == PrepConfigSetType.Combination:
            frame_config_set = OmegaConf.create({
                'modality': [Modality.Frame],
                'num_frame': [16],
                'sampling_method': ['uniform']
            })
            visiontoken_configs_set = OmegaConf.create({
                'modality': [Modality.VisionToken],
            })
            visiontoken_configs_set = OmegaConf.merge(frame_config_set, visiontoken_configs_set)
            ocr_configs_set = OmegaConf.create({
                'modality': [Modality.OCR],
            })
            ocr_configs_set = OmegaConf.merge(frame_config_set, ocr_configs_set)
            # asr_configs_set = OmegaConf.create({
            #     'modality': [Modality.ASR],
            #     'model': ['whisper-large'],
            #     'chunk_length': [3]
            # })
            caption_configs_set = OmegaConf.create({
                'modality': [Modality.Caption],
                'seconds_per_caption': [2],
                'frames_per_caption': [4],
                'model': [CaptionModelName.lavila]
            })
            object_configs_set = OmegaConf.create({
                'modality': [Modality.Object],
                'num_frame': [16],
                'confidence_threshold': [0.3],
                'model': [ObjModelName.APE_L_D],
            })
            if mode != 'preprocess':
                object_configs_set['retrieve_threshold'] = [0.26]
            clip_configs_set = OmegaConf.create({
                'modality': [Modality.Clip],
            })
            clip_configs_set = OmegaConf.merge(frame_config_set, clip_configs_set)

            modality2config_set = {
                Modality.Frame: frame_config_set,
                Modality.OCR: ocr_configs_set,
                # Modality.ASR: asr_configs_set,
                Modality.VisionToken: visiontoken_configs_set,
                Modality.Caption: caption_configs_set,
                Modality.Clip: clip_configs_set,
                Modality.Object: object_configs_set,
            }
        elif type == PrepConfigSetType.CaptionOnly:
            caption_configs_set = OmegaConf.create({
                'modality': [Modality.Caption],
                'seconds_per_caption': [1, 2, 4],
                'frames_per_caption': [1, 2, 4],
                'model': CaptionModelName.all_models
            })

            modality2config_set = {
                Modality.Caption: caption_configs_set,
            }
        elif type == PrepConfigSetType.VisiontokenOnly:
            frame_config_set = OmegaConf.create({
                'modality': [Modality.Frame],
                'num_frame': [4, 8, 16, 32, 64],
                'sampling_method': ['uniform', 'keyframe'],
            })
            visiontoken_configs_set = OmegaConf.create({
                'modality': [Modality.VisionToken],
            })
            visiontoken_configs_set = OmegaConf.merge(frame_config_set, visiontoken_configs_set)
            modality2config_set = {
                Modality.Frame: frame_config_set,
                Modality.VisionToken: visiontoken_configs_set,
            }
        elif type == PrepConfigSetType.ObjectOnly:
            object_config = {
                'modality': [Modality.Object],
                'num_frame': [4, 8, 16, 32],
                'confidence_threshold': [0.3, 0.4, 0.5],
                'model': ObjModelName.all_models,
                # 'retrieve_threshold': [0.26],
                # 'text_prompt': [None],
            }
            modality2config_set = {
                Modality.Object: object_config,
            }
        elif type == PrepConfigSetType.All:
            frame_config_set = OmegaConf.create({
                'modality': [Modality.Frame],
                'num_frame': [4, 8, 16, 32, 64],
                'sampling_method': ['uniform', 'keyframe'],
            })
            visiontoken_configs_set = OmegaConf.create(
                {
                    'modality': [Modality.VisionToken],
                    'num_frame': [4, 8, 16, 32, 64],
                    'sampling_method': ['uniform', 'keyframe'],
                }
            )
            ocr_configs_set = OmegaConf.create({
                'modality': [Modality.OCR],
            })
            ocr_configs_set = OmegaConf.merge(frame_config_set, ocr_configs_set)
            asr_configs_set = OmegaConf.create({
                'modality': [Modality.ASR],
                'model': ASRModelName.all_models,
                'chunk_length': [3, 6, 9]
            })
            caption_configs_set = OmegaConf.create({
                'modality': [Modality.Caption],
                'seconds_per_caption': [1, 2, 4],
                'frames_per_caption': [1, 2, 4],
                'model': CaptionModelName.all_models
            })
            object_configs_set = OmegaConf.create({
                'modality': [Modality.Object],
                'num_frame': [4, 8, 16, 32],
                'confidence_threshold': [0.3, 0.4, 0.5],
                'model': ObjModelName.all_models,
            })
            if mode != 'preprocess':
                object_configs_set['retrieve_threshold'] = [0.26]
            clip_configs_set = OmegaConf.create({
                'modality': [Modality.Clip],
                'num_frame': [16],
                'sampling_method': ['uniform'],
            })

            modality2config_set = {
                Modality.Frame: frame_config_set,
                Modality.OCR: ocr_configs_set,
                Modality.ASR: asr_configs_set,
                Modality.VisionToken: visiontoken_configs_set,
                Modality.Caption: caption_configs_set,
                Modality.Clip: clip_configs_set,
                Modality.Object: object_configs_set,
            }
        else:
            raise NotImplementedError

        return modality2config_set

    def groundqa_prepconfig(self, type, mode='preprocess'):
        if type == PrepConfigSetType.Combination:
            frame_config_set = OmegaConf.create({
                'modality': [Modality.Frame],
                'num_frame': [16],
                'sampling_method': ['uniform']
            })
            visiontoken_configs_set = OmegaConf.create({
                'modality': [Modality.VisionToken],
            })
            visiontoken_configs_set = OmegaConf.merge(frame_config_set, visiontoken_configs_set)
            ocr_configs_set = OmegaConf.create({
                'modality': [Modality.OCR],
            })
            ocr_configs_set = OmegaConf.merge(frame_config_set, ocr_configs_set)
            # asr_configs_set = OmegaConf.create({
            #     'modality': [Modality.ASR],
            #     'model': ['whisper-large'],
            #     'chunk_length': [3]
            # })
            caption_configs_set = OmegaConf.create({
                'modality': [Modality.Caption],
                'seconds_per_caption': [2],
                'frames_per_caption': [4],
                'model': [CaptionModelName.lavila]
            })
            object_configs_set = OmegaConf.create({
                'modality': [Modality.Object],
                'num_frame': [16],
                'confidence_threshold': [0.3],
                'model': [ObjModelName.APE_L_D],
            })
            if mode != 'preprocess':
                object_configs_set['retrieve_threshold'] = [0.26]
            clip_configs_set = OmegaConf.create({
                'modality': [Modality.Clip],
            })
            clip_configs_set = OmegaConf.merge(frame_config_set, clip_configs_set)

            modality2config_set = {
                Modality.Frame: frame_config_set,
                Modality.OCR: ocr_configs_set,
                # Modality.ASR: asr_configs_set,
                Modality.VisionToken: visiontoken_configs_set,
                Modality.Caption: caption_configs_set,
                Modality.Clip: clip_configs_set,
                Modality.Object: object_configs_set,
            }
        elif type == PrepConfigSetType.CaptionOnly:
            caption_configs_set = OmegaConf.create({
                'modality': [Modality.Caption],
                'seconds_per_caption': [30, 60, 90],
                'frames_per_caption': [1, 2, 4],
                'model': CaptionModelName.all_models
            })

            modality2config_set = {
                Modality.Caption: caption_configs_set,
            }
        elif type == PrepConfigSetType.VisiontokenOnly:
            frame_config_set = OmegaConf.create({
                'modality': [Modality.Frame],
                'num_frame': [4, 8, 16, 32, 64],
                'sampling_method': ['uniform', 'keyframe'],
            })
            visiontoken_configs_set = OmegaConf.create({
                'modality': [Modality.VisionToken],
            })
            visiontoken_configs_set = OmegaConf.merge(frame_config_set, visiontoken_configs_set)
            modality2config_set = {
                Modality.Frame: frame_config_set,
                Modality.VisionToken: visiontoken_configs_set,
            }
        elif type == PrepConfigSetType.ObjectOnly:
            object_config = {
                'modality': [Modality.Object],
                # 'num_frame': [4, 8, 16, 32],
                'num_frame': [4, 8, 16],
                'confidence_threshold': [0.3, 0.4, 0.5],
                'model': ObjModelName.all_models,
                # 'retrieve_threshold': [0.26],
                # 'text_prompt': [None],
            }
            modality2config_set = {
                Modality.Object: object_config,
            }
        elif type == PrepConfigSetType.All:
            frame_config_set = OmegaConf.create({
                'modality': [Modality.Frame],
                'num_frame': [4, 8, 16, 32, 64],
                # 'sampling_method': ['uniform', 'keyframe'],
                'sampling_method': ['uniform'],
            })
            visiontoken_configs_set = OmegaConf.create(
                {
                    'modality': [Modality.VisionToken],
                    'num_frame': [4, 8, 16, 32, 64],
                    # 'sampling_method': ['uniform', 'keyframe'],
                    'sampling_method': ['uniform'],
                }
            )
            ocr_configs_set = OmegaConf.create({
                'modality': [Modality.OCR],
            })
            ocr_configs_set = OmegaConf.merge(frame_config_set, ocr_configs_set)
            asr_configs_set = OmegaConf.create({
                'modality': [Modality.ASR],
                'model': ASRModelName.all_models,
                'chunk_length': [30, 60, 90]
            })
            caption_configs_set = OmegaConf.create({
                'modality': [Modality.Caption],
                'seconds_per_caption': [30, 60, 90],
                'frames_per_caption': [1, 2, 4],
                'model': CaptionModelName.all_models
            })
            object_configs_set = OmegaConf.create({
                'modality': [Modality.Object],
                # 'num_frame': [4, 8, 16, 32],
                'num_frame': [4, 8, 16],
                'confidence_threshold': [0.3, 0.4, 0.5],
                'model': ObjModelName.all_models,
            })
            if mode != 'preprocess':
                object_configs_set['retrieve_threshold'] = [0.26]
            clip_configs_set = OmegaConf.create({
                'modality': [Modality.Clip],
                'num_frame': [16],
                'sampling_method': ['uniform'],
            })

            modality2config_set = {
                Modality.Frame: frame_config_set,
                Modality.OCR: ocr_configs_set,
                Modality.ASR: asr_configs_set,
                Modality.VisionToken: visiontoken_configs_set,
                Modality.Caption: caption_configs_set,
                Modality.Clip: clip_configs_set,
                Modality.Object: object_configs_set,
            }
        else:
            raise NotImplementedError

        return modality2config_set

    def allvb_prepconfig(self, type, mode='preprocess'):
        if type == PrepConfigSetType.Combination:
            frame_config_set = OmegaConf.create({
                'modality': [Modality.Frame],
                'num_frame': [16],
                'sampling_method': ['uniform']
            })
            visiontoken_configs_set = OmegaConf.create({
                'modality': [Modality.VisionToken],
            })
            visiontoken_configs_set = OmegaConf.merge(frame_config_set, visiontoken_configs_set)
            ocr_configs_set = OmegaConf.create({
                'modality': [Modality.OCR],
            })
            ocr_configs_set = OmegaConf.merge(frame_config_set, ocr_configs_set)
            asr_configs_set = OmegaConf.create({
                'modality': [Modality.ASR],
                'model': ['whisper-large'],
                'chunk_length': [3]
            })
            caption_configs_set = OmegaConf.create({
                'modality': [Modality.Caption],
                'seconds_per_caption': [8],
                'frames_per_caption': [4],
                'model': [CaptionModelName.lavila]
            })
            ape_configs = OmegaConf.load(Path(__file__).parent / 'ape.yaml').model2config
            object_configs_set = OmegaConf.create({
                'modality': [Modality.Object],
                'num_frame': [16],
                'confidence_threshold': [0.3],
                'model': [ObjModelName.APE_L_D],
                # 'retrieve_threshold': [0.26],
                # 'text_prompt': [None],
            })
            if mode != 'preprocess':
                object_configs_set['retrieve_threshold'] = [0.26]
            clip_configs_set = OmegaConf.create({
                'modality': [Modality.Clip],
            })
            clip_configs_set = OmegaConf.merge(frame_config_set, clip_configs_set)
            modality2config_set = {
                Modality.Frame: frame_config_set,
                Modality.OCR: ocr_configs_set,
                Modality.ASR: asr_configs_set,
                Modality.VisionToken: visiontoken_configs_set,
                Modality.Caption: caption_configs_set,
                Modality.Clip: clip_configs_set,
                Modality.Object: object_configs_set,
            }
        elif type == PrepConfigSetType.VisiontokenOnly:
            frame_config_set = OmegaConf.create({
                'modality': [Modality.Frame],
                'num_frame': [4, 8, 16, 32, 64],
                'sampling_method': ['uniform', 'keyframe'],
            })
            visiontoken_configs_set = OmegaConf.create({
                'modality': [Modality.VisionToken],
            })
            visiontoken_configs_set = OmegaConf.merge(frame_config_set, visiontoken_configs_set)
            modality2config_set = {
                Modality.Frame: frame_config_set,
                Modality.VisionToken: visiontoken_configs_set,
            }
        elif type == PrepConfigSetType.CaptionOnly:
            caption_configs_set = OmegaConf.create({
                'modality': [Modality.Caption],
                'seconds_per_caption': [32, 64, 96, 128],
                'frames_per_caption': [1, 2, 4],
                'model': CaptionModelName.all_models
            })
            modality2config_set = {
                Modality.Caption: caption_configs_set,
            }
        else:
            raise NotImplementedError
        return modality2config_set

    def agqa_prepconfig(self):
        raise NotImplementedError

    def cinepipe_prepconfig(self):
        raise NotImplementedError


# %%
if __name__ == '__main__':
    loader = VideoDatasetLoader()
    video_id2questions, video_dir, question_type, video_id2name = loader.load_dataset(VQADataset.GroundQA,
                                                                                      top_k_video=120)
    print(video_id2questions)
