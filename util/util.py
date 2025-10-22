import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from types import NoneType

import torch  # 一定要先加载torch后导入numpy
import numpy as np
from decord import VideoReader, cpu
import random

import socket
import pickle

from markdown_it.common.html_re import processing
from omegaconf import OmegaConf
import os
import json

from preprocess_constant import modality2priority, PrepConfigSetType, Modality, CaptionModelName, KeyWordExtractorNames, \
    QuestionType, LLMName, KnobType
import time
from sklearn.metrics.pairwise import cosine_similarity
import optuna
import optunahub

id2category = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}

LLM_INF = int(1e5)


def first_char_as_answer(res, num_choice=5):
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    if len(res) == 0:
        return -1
    if res[0] in mapping:
        return mapping[res[0]]
    return -1


def first_char_as_answer_raw(res, num_choice=5):
    candidates = ['A', 'B', 'C', 'D', 'E']
    if len(res) == 0:
        return random.choice(candidates[0:num_choice])
    if res[0] in candidates:
        return res[0]
    return random.choice(candidates[0:num_choice])


def identity(res):
    return res


def first_char_after_anchor(anchor):
    def f(res):
        mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        anchor_index = res.find(anchor)
        pred = -1  # if decoding failed, return -1
        if anchor_index >= 0:
            pred_letter = res[anchor_index + len(anchor)]
            if pred_letter in mapping:
                pred = mapping[pred_letter]
        return pred

    return f


def process_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames, frame_time, video_time


def sample_video_frames(video_path, num_frame):
    vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
    total_frame_num = len(vr)
    duration = total_frame_num / vr.get_avg_fps()
    interval = round(total_frame_num / num_frame)
    frame_idxs = [i for i in range(0, len(vr), interval)]
    frame_time = [i / vr.get_avg_fps() for i in frame_idxs]
    # frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idxs).asnumpy()

    return spare_frames[0:num_frame], frame_idxs[0:num_frame]


def config2path_load(config):  # config转化为路径字符串
    escape_keys = ['modality', 'frame_idx', 'text_prompt']
    return config2path(config, escape_keys)


def config2path(config, escape_keys):
    assert len(config) > 0
    config_str = ''
    for key, val in config.items():
        if key in escape_keys:
            continue
        config_str += config_name2abbreviation(key) + '-' + str(val) + '|'
    config_str = config_str[:-1]
    return config_str


def config2path_preprocess(config):  # config转化为路径字符串
    escape_keys = ['modality', 'retrieve_threshold']
    return config2path(config, escape_keys)


def preprocess_store_path(modality, config, video_name, dataset_name, llm_name=None):
    home_dir = os.getenv("HOME")
    if config is not None:  # modality
        store_dir = os.path.join(home_dir, 'preprocessing', dataset_name, video_name, modality)
        if modality == Modality.VisionToken:
            assert llm_name is not None
            store_dir = os.path.join(home_dir, 'preprocessing', dataset_name, video_name, f'{modality}_{llm_name}')
        os.makedirs(store_dir, exist_ok=True)
        store_path = os.path.join(store_dir, config2path_preprocess(config))
    else:  # video meta信息
        store_dir = os.path.join(home_dir, 'preprocessing', dataset_name, video_name)
        os.makedirs(store_dir, exist_ok=True)
        store_path = store_dir
    return store_path


def config_name2abbreviation(config_name: str):
    return ''.join([word[0] for word in config_name.split('_')])


# def get_abbr_config(config : dict):
#     abbr_config = {}
#     escape_keys = ['modality']
#     for key, value in config.items():
#         if key in escape_keys:
#             continue
#         # if key == 'text_prompt':
#         #     if value is not None:
#         #         abbr_config[config_name2abbreviation(key)] = 'Dynamic'
#         #     else:
#         #         abbr_config[config_name2abbreviation(key)] = value
#         #     continue
#         abbr_config[config_name2abbreviation(key)] = value
#     return abbr_config


def extract_frame_config(config):
    return OmegaConf.create({
        'num_frame': config['num_frame'],
        'sampling_method': config['sampling_method']
    })


def get_inference_result_file_name(modality_config: dict):
    modalities = list(modality_config.keys())
    modalities = sorted(modalities, key=lambda x: modality2priority[x])
    inference_result_file_name = ''
    for modality in modalities:
        config_path = config2path_load(modality_config[modality])
        if inference_result_file_name == '':
            inference_result_file_name = '_'.join([modality, config_path])
        else:
            inference_result_file_name = '_'.join([inference_result_file_name, modality, config_path])
    return inference_result_file_name


def has_retrieved_modality(modality_config):
    for modality in modality_config.keys():
        if modality in Modality.RetrieveModalities:
            return True
    return False


def get_inference_result_dir(llm_name, dataset_name, video_id, modality_config: dict, keyword_extractor_name):
    # modalities = list(modality_config.keys())
    # modalities = sorted(modalities, key=lambda x: modality2priority[x])
    # inference_result_file_name = ''
    # for modality in modalities:
    #     config_path = config2path_load(modality_config[modality])
    #     if inference_result_file_name == '':
    #         inference_result_file_name = '_'.join([modality, config_path])
    #     else:
    #         inference_result_file_name = '_'.join([inference_result_file_name, modality, config_path])

    # if not has_retrieved_modality(modality_config):  # todo 这里默认用LLM extractor作为不需要retrieve的modality
    #     keyword_extractor_name = KeyWordExtractorNames.LLMExtractor
    inference_result_file_name = get_inference_result_file_name(modality_config)
    inference_result_dir = os.path.join(os.getenv('HOME'), 'vqa_result', llm_name, dataset_name, 'inference_result',
                                        video_id,
                                        keyword_extractor_name)
    os.makedirs(inference_result_dir, exist_ok=True)
    inference_result_dir = os.path.join(inference_result_dir, inference_result_file_name)
    return inference_result_dir


def get_video_path(dataset_name, video_name):
    return os.path.join(os.getenv("HOME"), 'dataset', dataset_name, 'videos', video_name)


def caption2prompt(caption: dict):  # 按照LLoVi的方法把caption转化成输入的prompt
    caption_prompt = []
    for i, cap in enumerate(caption.values()):
        caption_prompt.append(f'{i}: {cap}\n')
    return caption_prompt


def set_random_seed(seed=42):
    global np_state
    global rd_state
    random.seed(seed)
    np.random.seed(seed)
    np_state = np.random.get_state()
    rd_state = random.getstate()
    torch.manual_seed(seed)
    print(f'set random seed {seed}')


def send_close_server_request(modality):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('0.0.0.0', server_config[modality]['port']))
    print(f'Closing server socket {modality}')
    if modality == Modality.VisionToken or modality == Modality.Object:
        data = (None, None, None, None)
    else:
        data = (None, None, None)
    client_socket.send(pickle.dumps(data))


def get_video_meta_path(video_name, dataset_name):
    return os.path.join(preprocess_store_path(None, None, video_name, dataset_name), 'video_meta')


def abbr_to_config(abbrs):
    abbr_dict = {
        'nf': 'num_frame',
    }
    if isinstance(abbrs, str):
        return abbr_dict[abbrs]
    elif isinstance(abbrs, list):
        return [abbr_dict[abbr] for abbr in abbrs]
    else:
        raise TypeError


def merge_configs():
    pass


def get_retrieve_file_name(modality, config=None):
    if config is not None:
        file_name = f'{modality}_retrieve_result_{config2path_load(config)}.json'
    else:
        file_name = f'{modality}_retrieve_result.json'
    return file_name


def load_retrieve_result(llm_name, dataset_name, video_id, modality, keyword_extractor, config=None):
    retrieve_result_dir = os.path.join(os.getenv('HOME'), 'vqa_result', llm_name, dataset_name, 'retrieve_results',
                                       video_id,
                                       keyword_extractor.name)
    os.makedirs(retrieve_result_dir, exist_ok=True)
    file_name = get_retrieve_file_name(modality, config)
    retrieve_result_file = os.path.join(retrieve_result_dir, file_name)
    if os.path.exists(retrieve_result_file):
        return json.load(open(retrieve_result_file, 'r'))
    else:
        return {}


def get_keyword_file_name(keyword_extractor):
    file_name = 'keyword_result.json'
    return file_name


def save_keyword_result(llm_name, keyword_result, dataset_name, video_id, keyword_extractor):
    retrieve_result_dir = os.path.join(os.getenv('HOME'), 'vqa_result', llm_name, dataset_name, 'retrieve_results',
                                       video_id,
                                       keyword_extractor.name)
    os.makedirs(retrieve_result_dir, exist_ok=True)
    file_name = get_keyword_file_name( keyword_extractor.name)
    retrieve_result_file = os.path.join(retrieve_result_dir, file_name)
    json.dump(keyword_result, open(retrieve_result_file, 'w'))


def load_keyword_result(llm_name, dataset_name, video_id, keyword_extractor):
    retrieve_result_dir = os.path.join(os.getenv('HOME'), 'vqa_result', llm_name, dataset_name, 'retrieve_results',
                                       video_id,
                                       keyword_extractor.name)
    os.makedirs(retrieve_result_dir, exist_ok=True)
    file_name = get_keyword_file_name(keyword_extractor.name)
    retrieve_result_file = os.path.join(retrieve_result_dir, file_name)
    if os.path.exists(retrieve_result_file):
        return json.load(open(retrieve_result_file, 'r'))
    else:
        return {}


def extract_ttfts(results, modalities, keyword_extractor_name, ignore_object=False):
    ttfts_e2e = []  # 包括keyword extraction retrieve以及llm 的ttft
    ttfts_llm = []  # 只有llm的ttft
    for result in results:
        ttft = 0
        if any([modality in Modality.RetrieveModalities for modality in modalities]):
            ttft += result['llm_retrieve_time'] if result.get('llm_retrieve_time') else result[
                'keyword_extraction_time']
        for key in result.keys():
            key_ = key.split('_')[0]
            if key_ in Modality.LoadModalities:
                if ignore_object and key_ == Modality.Object:
                    ttft += 0.0
                # elif not ignore_object and key_ == Modality.Object:
                # ttft += 0.005 # videoagent的measure结果
                # ttft += result['scene_graph_process_time'] if result.get('scene_graph_process_time') else 0.0
                else:
                    ttft += result[key] if result[key] else 0
        assert result['ttft'] > 0
        ttft += result['ttft']
        ttfts_e2e.append(ttft)
        ttfts_llm.append(result['ttft'])
    return ttfts_e2e, ttfts_llm


def extract_processing_time(llm_name, modality2config, video_name, dataset_name):
    total_processing_time = 0
    modalityconfig2processing_time = {}
    for modality, config in modality2config.items():
        if modality in Modality.PreprocessModalities:
            modality_path = preprocess_store_path(modality, config, video_name, dataset_name, llm_name)
            modality_process_result = pickle.load(open(modality_path, 'rb'))
            processing_time = modality_process_result['processing_time']
            total_processing_time += processing_time
            modalityconfig2processing_time[f'{modality}_{str(config)}'] = processing_time
    return total_processing_time, modalityconfig2processing_time


def extract_scores(results):
    if isinstance(results, dict):
        scores = []
        for result in results.values():
            scores.append(float(result[0]['score']))
    elif isinstance(results, list):
        scores = [result['score'] for result in results]
    else:
        raise NotImplementedError
    return scores


def extract_accuracy(results):
    if isinstance(results, dict):
        return results['acc']
    elif isinstance(results, list):
        binary_scores = [result['score'] for result in results]
        return sum(binary_scores) / len(results)
    else:
        raise NotImplementedError


def extract_accuracies(results):
    if isinstance(results, list):
        binary_scores = [result['score'] for result in results]
        return binary_scores
    else:
        raise NotImplementedError


def get_file2results(llm_name, dataset_name, video_id, selected_modality2config,
                     question_type, include_keyword_extraction_time=False,
                     keyword_extractor_name=''):
    # if not has_retrieved_modality(selected_modality2config[0]):
    #     keyword_extractor_name = KeyWordExtractorNames.LLMExtractor
    result_dir = os.path.join(os.getenv('HOME'), 'vqa_result', llm_name, dataset_name, 'inference_result', video_id,
                              keyword_extractor_name)
    file2eval_results = {}
    file2inference_results = {}
    file2config = {}

    if question_type == QuestionType.OE:
        selected_file_names = [get_inference_result_file_name(modality2config) for modality2config in
                               selected_modality2config]
        for file, config in zip(selected_file_names, selected_modality2config):
            file2config[file] = config
            file2eval_results[file] = json.load(
                open(os.path.join(result_dir, file + '_eval_result', 'eval_result.json'), 'r'))
            file2inference_results[file] = json.load(open(os.path.join(result_dir, file), 'r'))
    else:
        for modality2config in selected_modality2config:
            file = get_inference_result_file_name(modality2config)
            eval_result_dir = '_'.join(
                [get_inference_result_dir(dataset_name, video_id, modality2config, keyword_extractor_name),
                 'eval_result'])
            file2eval_results[file] = json.load(open(os.path.join(eval_result_dir, 'eval_result.json'), 'r'))
            file2inference_results[file] = json.load(open(os.path.join(result_dir, file), 'r'))
    return file2eval_results, file2inference_results, file2config


def vqa_train_test_split(eval_results, train_set_ratio, test_set_ratio=None, method='random'):
    assert isinstance(train_set_ratio, float)
    assert isinstance(test_set_ratio, (float, NoneType))
    num_ids = len(eval_results) if isinstance(eval_results, list) else eval_results
    ids = [i for i in range(num_ids)]
    if method == 'sequential':
        train_ids = [i for i in range(int(num_ids * train_set_ratio))]
    elif method == 'random':
        np.random.set_state(np_state)
        train_ids = np.random.choice(ids, int(num_ids * train_set_ratio), replace=False).tolist()
    else:
        raise NotImplementedError

    test_ids = [i for i in range(num_ids) if i not in train_ids]
    if len(test_ids) == 0:
        test_ids = train_ids
    if test_set_ratio is None:  # train 和 test 不相交
        if train_set_ratio != 1.0:
            assert all(test_id not in train_ids for test_id in test_ids)
        assert len(test_ids) + len(train_ids) == num_ids or len(test_ids) + len(train_ids) == len(
            eval_results) * 2
    else:
        assert test_set_ratio >= train_set_ratio
        if test_set_ratio == 1.0:
            test_ids = [i for i in range(num_ids)]
        else:
            test_ids = train_ids.copy()
            rest_ids = [id for id in range(num_ids) if id not in test_ids]
            rest_test_ids = np.random.choice(rest_ids, int(num_ids * (test_set_ratio - train_set_ratio)),
                                             replace=False).tolist()
            test_ids.extend(rest_test_ids)
    # train_eval_results = [eval_results[i] for i in train_ids]
    # test_eval_results = [eval_results[i] for i in test_ids]
    return train_ids, test_ids


def load_evaluate_result_for_one_video(**kwargs):
    dataset_name = kwargs['dataset_name']
    video_id = kwargs['video_id']
    video_name = kwargs['video_name']
    selected_modality2config = kwargs['selected_modality2config']
    question_type = kwargs['question_type']
    keyword_extractor_name = kwargs['keyword_extractor_name']
    llm_name = kwargs['llm_name']
    file2eval_results, file2inference_results, file2config = get_file2results(llm_name, dataset_name, video_id,
                                                                              selected_modality2config,
                                                                              question_type,
                                                                              include_keyword_extraction_time=False,
                                                                              keyword_extractor_name=keyword_extractor_name)
    eval_results = list(file2eval_results.values())[0]
    inference_results = list(file2inference_results.values())[0]
    if question_type == QuestionType.OE:
        eval_results = dict(sorted(eval_results.items(), key=lambda item: int(item[0].split('_')[-1])))
        eval_results = list(eval_results.values())
        for i, result in enumerate(inference_results):
            result['score'] = eval_results[i][0]['score']
    eval_results = inference_results
    if question_type == QuestionType.OE:
        whole_set_accuracies = extract_scores(eval_results)
    else:
        whole_set_accuracies = extract_accuracies(eval_results)
    modalties = list(selected_modality2config[0].keys())
    whole_set_e2elatencys, _ = extract_ttfts(eval_results, modalties, keyword_extractor_name)
    total_processing_time, modalityconfig2processing_time = extract_processing_time(llm_name, selected_modality2config[0],
                                                                                    video_name, dataset_name)

    return whole_set_accuracies, whole_set_e2elatencys, total_processing_time, modalityconfig2processing_time


def filename2modalities(file_name):
    modalities = []
    for modality in Modality.MainModalities:
        if modality in file_name:
            modalities.append(modality)
    return modalities


def validate_config(modality, configs):
    if modality == Modality.Caption:
        if configs['model'] == CaptionModelName.llava:
            configs['frames_per_caption'] = 1


def encode_sentences(sentence_list, model_name, sentence_model, tokenize_func):
    '''given a list of sentences, return the embeddings for them using the sentence encoder model'''
    assert model_name == 'clip'
    emb_list = []
    device = "cuda"
    num_iter = 1
    with torch.no_grad():
        for sentence in sentence_list:
            # emb_list.append(sentence_model.encode_text(clip.tokenize([sentence]).to(device)).cpu().numpy())
            emb_list.append(sentence_model.encode_text(tokenize_func([sentence]).to(device)).cpu().numpy())
    emb_list = np.concatenate(emb_list, axis=0)
    return emb_list


def retrieve_candidate_objects(description, uid2emb, uid2category, sentence_model, tokenize_func,
                               retrieve_threshold=0.26):
    def select_elements_with_repetition(lst, count):
        if not lst:  # 如果列表为空，返回空列表
            return []

        result = []
        for i in range(count):
            # 使用模运算来循环获取元素
            result.append(lst[i % len(lst)])
        return result

    def compute_cosine_similarity(target_embedding, embedding_list):
        embedding_list_np = [embedding.numpy() for embedding in embedding_list]
        if len(embedding_list_np) == 0:
            return np.array([])
        target_embedding_tensor = target_embedding.reshape(1, -1)
        # Compute cosine similarity
        start = time.monotonic()
        similarity_scores = cosine_similarity(target_embedding_tensor, embedding_list_np)
        end = time.monotonic()
        return similarity_scores.reshape(-1)

    des_emb = encode_sentences([f"a photo of a {description}."], model_name='clip', sentence_model=sentence_model,
                               tokenize_func=tokenize_func)
    scores = compute_cosine_similarity(des_emb, list(uid2emb.values()))
    indices = np.where(scores >= retrieve_threshold)[0]
    selected_categories = [uid2category[i] for i in indices]
    candidate_uids = []
    for i in indices:
        candidate_uids.append(list(uid2emb)[i])
    return candidate_uids, scores

    # main()


def get_top_configs_acc(config_acc_list, method='top_k', top_k=1, top_margin=0.1, latency_list=None):
    config_acc_list = sorted(config_acc_list, key=lambda x: -x[1])
    if method == 'top_k':
        top_list = config_acc_list[:top_k]
    elif method == 'top_margin':  # 比如在最高分5%差距以内的?
        top_list = []
        max_acc = config_acc_list[0][1]
        if latency_list is not None:
            for (config, acc), latency in zip(config_acc_list, latency_list):
                if 1 - acc / max_acc <= top_margin:
                    top_list.append((config, acc, latency))
                else:
                    break
        else:
            for config, acc in config_acc_list:
                if 1 - acc / max_acc <= top_margin:
                    top_list.append((config, acc))
                else:
                    break
    else:
        raise NotImplementedError
    return top_list


def get_retrieved_object_result(object_result, keywords, uid2clip, uid2category, config, tokenize_func, sentence_model,
                                use_org_label=False):
    # retrieve_threshold = 0.26
    retrieve_threshold = 0.26
    if config.get('retrieve_threshold'):
        retrieve_threshold = config['retrieve_threshold']
    start_time = time.monotonic()
    candidate_uids = set()
    uid2_keyword = {}
    if keywords is None:  # 返回所有的object 通过设置retrieve_threshold = 0 实现
        uid2_keyword = uid2category
        uids, retrieve_scores = retrieve_candidate_objects('object', uid2clip, uid2category, sentence_model,
                                                           tokenize_func, retrieve_threshold=0.0)
        candidate_uids.update(uids)
    else:
        for keyword in keywords:
            uids, retrieve_scores = retrieve_candidate_objects(keyword, uid2clip, uid2category, sentence_model,
                                                               tokenize_func, retrieve_threshold=retrieve_threshold)
            candidate_uids.update(uids)
            for uid in uids:
                uid2_keyword[uid] = keyword
    end_time = time.monotonic()
    retrieved_object_result = []
    for frame_result_str in object_result:
        frame_result = frame_result_str.strip().split(';')
        retrieved_frame_result = ''
        for object_str in frame_result:
            tmp = object_str.split(':')
            org_category_name, bbox = tmp[0], tmp[1]
            uid = int(org_category_name.split('_')[-1])
            if uid in candidate_uids:
                category_name = org_category_name.split('_')[0] if use_org_label else uid2_keyword[uid]
                retrieved_frame_result += f'{category_name} / {org_category_name}: {bbox}; '
        if len(retrieved_frame_result) > 0:
            retrieved_frame_result = retrieved_frame_result[:-2]
        if len(retrieved_frame_result) > 0:
            retrieved_object_result.append(retrieved_frame_result)
    retrieve_time = end_time - start_time
    return retrieved_object_result, retrieve_time


from pathlib import Path

# 获取当前脚本的绝对路径
current_file_path = Path(__file__).resolve()
print("当前脚本路径:", current_file_path)

# 获取脚本所在目录
project_path = current_file_path.parent
print("项目路径:", project_path)


def get_tuning_result_file_path(llm_name, sampler_name, dataset_name, video_id, init_sample_num, seed, train_set_ratio,
                                latency_constraint=None, extractor_name='LLM'):
    result_dir = f'{project_path}/tuning_results/{llm_name}/{sampler_name}/{dataset_name}/{extractor_name}'
    os.makedirs(result_dir, exist_ok=True)
    file_name = f'{video_id}_InitSample_{init_sample_num}_Seed_{seed}_TrainRatio_{train_set_ratio}'
    if latency_constraint is not None:
        file_name += f'_Constraint_{latency_constraint}'
    file_path = file_name + '.pkl'
    file_path = os.path.join(project_path, result_dir, file_path)
    result_key = f'{llm_name}_{sampler_name}_{dataset_name}_{extractor_name}_{file_name}'
    return file_path, result_key, result_dir


def get_visiontokensampler_config(modality2config_set):
    configs = []
    for num_frame in [4, 8, 16, 32, 64]:
        sampling_method = 'uniform'
        config = {
            Modality.VisionToken: {
                'num_frame': num_frame,
                'sampling_method': sampling_method,
            }
        }
        configs.append(config2optuna_config(config, modality2config_set))
    return configs


def sampler_factory(sampler_name, init_sample_num, seed, constraints_func, **kwargs):
    if sampler_name == 'RandomSampler':
        return optuna.samplers.RandomSampler(seed=seed), None
    elif sampler_name == 'NSGAIISampler':
        population_size = kwargs['population_size']
        return optuna.samplers.NSGAIISampler(seed=seed, constraints_func=constraints_func,
                                             population_size=population_size), None
    elif sampler_name == 'TPESampler':
        if constraints_func is None:
            return optuna.samplers.TPESampler(seed=seed, n_startup_trials=init_sample_num,
                                              constraints_func=None), None
        else:
            return optunahub.load_module(package="samplers/ctpe").cTPESampler(seed=seed,
                                                                              n_startup_trials=init_sample_num,
                                                                              constraints_func=constraints_func), None
    elif sampler_name == 'GPSampler':
        return optuna.samplers.GPSampler(seed=seed, n_startup_trials=init_sample_num,
                                         constraints_func=constraints_func), None
    elif sampler_name == 'VisionTokenSampler':
        modality2config_set = kwargs['modality2config_set']
        return optuna.samplers.RandomSampler(seed=seed), get_visiontokensampler_config(modality2config_set)
    else:
        raise NotImplementedError
def estimate_token_upper_bound():
    return 0
def load_01acc():
    pass
def load_tuning_result(llm_name, dataset_name, video_id2questions, samplers, seeds, init_sample_num=5,
                       latency_constraints=[None],
                       train_set_ratio=0.5, extractor_name='LLM'):
    filename2results = {}
    for video_id in video_id2questions.keys():
        for sampler in samplers:
            if sampler == 'VLMTuner':
                init_sample_num = 0
            else:
                init_sample_num = 5
            for seed in seeds:
                for latency_constraint in latency_constraints:
                    file_path, result_key, _ = get_tuning_result_file_path(llm_name, sampler, dataset_name, video_id,
                                                                           init_sample_num, seed, train_set_ratio,
                                                                           latency_constraint, extractor_name)
                    if os.path.exists(file_path):
                        result = pickle.load(open(file_path, 'rb'))
                        filename2results[result_key] = result
                        result['whole_set_01acc_history'] = []
                        for trial in result['trail_history']:
                            selected_modality2config = trial['selected_modality2config']
                            question_type = QuestionType.OE
                            try:
                                file2eval_results, file2inference_results, file2config = get_file2results(llm_name,
                                                                                                      dataset_name,
                                                                                                      video_id,
                                                                                                      selected_modality2config,
                                                                                                      question_type,
                                                                                                      include_keyword_extraction_time=False,
                                                                                                      keyword_extractor_name=extractor_name)
                            except FileNotFoundError:
                                result['whole_set_01acc_history'].append([0] * (len(result['train_ids']) + len(result['test_ids'])))
                                continue
                            id_01acc = [(int(key.split('_')[-1]), val[0]['pred']) for key,val in list(file2eval_results.values())[0].items()]
                            id_01acc.sort(key=lambda x: x[0])
                            result['whole_set_01acc_history'].append([
                                0 if item[1] == 'no' else 1 for item in id_01acc
                            ])
                        actual_len = min([len(item) for item in result['whole_set_01acc_history']])
                        final_01acc= []
                        for item in result['whole_set_01acc_history']:
                            item = item[0:actual_len]
                            final_01acc.append(item)
                        result['whole_set_01acc_history'] = final_01acc
    return filename2results


def performance_improvement(performance_list, latency_list=None, latency_constraint=None):
    best_performance = 0
    prev_list = None
    improvement_list = []
    if latency_list is None:
        latency_list = [1e5] * len(performance_list)
    for performance, latency in zip(performance_list, latency_list):
        if isinstance(performance, list):
            if latency_constraint is None or latency <= latency_constraint:
                if prev_list is None:
                    best_performance = np.mean(performance)
                    prev_list = performance
                else:
                    prev_list = np.maximum(np.array(performance), np.array(prev_list))
                    best_performance = np.mean(prev_list)
        else:
            if performance > best_performance:
                if latency_constraint is None or (latency <= latency_constraint):
                    best_performance = performance
        improvement_list.append(best_performance)
    return improvement_list


def config2optuna_config(config, modality2config_set):
    modality2use_modality_knob_names = {modality: f'use_{modality}' for modality in
                                        modality2config_set.keys()}
    optuna_config = {}
    for modality, knob_name in modality2use_modality_knob_names.items():
        if modality in Modality.NonLLMModalities:
            continue
        if modality in config.keys():
            optuna_config[knob_name] = True
        else:
            optuna_config[knob_name] = False
    for modality, config2value in config.items():
        for config, value in config2value.items():
            optuna_config[get_optuna_knob_name(modality, config)] = value
    return optuna_config


def get_optuna_knob_name(modality: str, config_name: str):
    return f'{modality}_{config_name}'


def lhs_sampling(space, n_samples, seed=None):
    """
    同时考虑所有 model 的 knob 做 LHS 采样，返回 dict 格式。

    返回结构:
    {
      0: {"model_A": {...}, "model_B": {...}},
      1: {"model_A": {...}, "model_B": {...}},
      ...
    }
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 展平所有 knobs
    flat_knobs = {}
    for mname, knobs in space.items():
        for kname, values in knobs.items():
            flat_knobs[(mname, kname)] = values

    knob_keys = list(flat_knobs.keys())
    samples = []

    for key in knob_keys:
        values = flat_knobs[key]

        # 连续变量
        if len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
            low, high = values
            cut = np.linspace(0, 1, n_samples + 1)
            u = np.random.rand(n_samples)
            points = low + (cut[:-1] + u * (cut[1:] - cut[:-1])) * (high - low)
            np.random.shuffle(points)

        # 离散 / categorical
        else:
            unique_vals = list(values)
            m = len(unique_vals)

            if m >= n_samples:
                points = random.sample(unique_vals, n_samples)
            else:
                points = unique_vals.copy()
                extra = random.choices(unique_vals, k=n_samples - m)
                points.extend(extra)
                random.shuffle(points)

        samples.append(points)

    # 转置并还原成 model-结构
    samples = np.array(samples, dtype=object).T
    results = {}
    for i, s in enumerate(samples):
        config = {}
        for (mname, kname), val in zip(knob_keys, s):
            config.setdefault(mname, {})[kname] = val
        results[i] = config

    return results


class InferenceParameter:
    def __init__(self, llm_name=LLMName.LlavaVideoQwen7b):
        self.use_existing_retrieve_result = False
        self.use_existing_keyword_result = False
        self.do_inference = True
        self.need_save_inference_result = True
        self.need_save_retrieve_result = True
        self.need_save_keyword_result = True
        self.force_inference = True
        self.llm_name = llm_name


def avg_set_similarity(set1):  # 衡量一个set里元素的
    sim_matrix = np.cosine_similarity(set1)
    pass

def judge_knob_type(knob_choice):
    assert len(knob_choice) > 0
    if len(knob_choice) == 1:
        return KnobType.CategoricalKnob
    if isinstance(knob_choice[0], (str, bool)):
        return KnobType.CategoricalKnob
    is_categorical = False
    step = knob_choice[1] - knob_choice[0]
    for i in range(1, len(knob_choice) - 1):
        if knob_choice[i] + step != knob_choice[i + 1]:
            is_categorical = True
    if is_categorical:
        return KnobType.CategoricalKnob
    if isinstance(knob_choice[0], float):
        return KnobType.FloatKnob
    if isinstance(knob_choice[0], int):
        return KnobType.IntKnob

np_state = None
rd_state = None
default_seed = 42
set_random_seed(default_seed)
server_config = OmegaConf.load(os.path.join(os.getenv('HOME'), 'code/Video-RAG-master-main', 'server_config.yaml'))
warm_up_video_name = '2440175990.mp4'
warm_up_dataset = 'sample_video'
warm_up_video_path = os.path.join(os.getenv('HOME'), 'dataset', warm_up_dataset, 'videos', warm_up_video_name)
hatch = ['//', '\\\\', '||', '-', '+', None, '.', '*', 'x', 'o', 'O', ]
token_margin = 100
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

last_video_num = 120  # 上一次inference的video数量
