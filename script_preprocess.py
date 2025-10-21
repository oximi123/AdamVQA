if __name__ != "__main__":
    raise ImportError("This module can only be run directly, not imported!")
import pickle

from preprocess_config import *
from load_dataset import VideoDatasetLoader

def send_preprocess_request(llm_name, dataset_name, video_name, config, results=None):
    print('Config: {}'.format(config))
    modality = config['modality']
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('0.0.0.0', server_config[modality]['port']))
    if config['modality'] == Modality.Object:
        data = (dataset_name, video_name, config, None)
    elif config['modality'] == Modality.VisionToken:
        data = (dataset_name, video_name, config, llm_name)
    else:
        data = (dataset_name, video_name, config)
    client_socket.send(pickle.dumps(data))
    result_path = pickle.loads(client_socket.recv(4096))
    result = pickle.load(open(result_path, 'rb'))
    if results is not None:
        results.append(result)
    return result


if __name__ == '__main__':
    # video_name = '2440175990.mp4'
    # dataset_name = 'sample_video'
    llm_name = LLMName.LlavaVideoQwen7b # todo
    # llm_name = LLMName.VideoLlamaMistral7b
    top_k_video = 200
    # dataset_name = VQADataset.MSVD_QA # todo
    # dataset_name = VQADataset.MSRVTT_QA
    dataset_name = VQADataset.GroundQA
    # dataset_name = VQADataset.SampleVideo
    dataset_loader = VideoDatasetLoader()
    config_set_type = PrepConfigSetType.All
    video_id2questions, dataset_path, question_type, video_id2name = dataset_loader.load_dataset(dataset_name,
                                                                                                 top_k_video=top_k_video)
    modality2config_set = dataset_loader.get_dataset_prep_config_set(dataset_name, type=config_set_type)
    for modality, configs in modality2config_set.items(): # 枚举每一个modality
        modality2results = {
            modality: [] for modality in modality2config_set.keys()
        }
        all_configs = PrepConfig.generate_config_combinations(configs, modality) #针对modality的所有config做组合
        for config_key in all_configs[0].keys():
            if 'model' in config_key:
                all_configs = sorted(all_configs, key=lambda config: config['model'])
                break
        for config in all_configs: # 枚举每一个config
            for video_id in video_id2questions.keys():
                video_name = video_id2name[video_id]
                print(f'--------------------------{video_name}--------------------------')
                video_path = get_video_path(dataset_name, video_name)
                vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
                total_frame_num = len(vr)
                duration = total_frame_num / vr.get_avg_fps()
                video_meta = {
                    'duration': duration,
                    'fps': vr.get_avg_fps()
                }
                pickle.dump(video_meta, open(get_video_meta_path(video_name, dataset_name), 'wb'))
                close_server_after_processing = True
                result = send_preprocess_request(llm_name, dataset_name, video_name, OmegaConf.create(config), modality2results[modality])
                print(result)
        if close_server_after_processing:
            send_close_server_request(modality)
# %% visiontoken

# %% object

# %% asr


# %% ocr

# %%
