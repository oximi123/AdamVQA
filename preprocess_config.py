import itertools

from omegaconf import OmegaConf, DictConfig
from preprocess_constant import *
from util import *
base_modalities = [Modality.Caption, Modality.VisionToken]
optional_modalities = [Modality.Caption, Modality.Object, Modality.ASR, Modality.OCR]


# frame
frame_config_set = OmegaConf.create({
    'modality': ['frame'],
    # 'num_frame': [4, 8, 16, 32, 64],
    'num_frame': [16],
    # 'sampling_method': ['uniform', 'keyframe']
    'sampling_method': ['uniform']
})

# visiontoken
visiontoken_configs_set = OmegaConf.create({
    'modality': ['visiontoken'],
})
visiontoken_configs_set = OmegaConf.merge(frame_config_set, visiontoken_configs_set)

# ocr
ocr_configs_set = OmegaConf.create({
    'modality': ['ocr'],
})
ocr_configs_set = OmegaConf.merge(frame_config_set, ocr_configs_set)

# asr
asr_configs_set = OmegaConf.create({
    'modality': ['asr'],
    'model': ['whisper-large'],
    # 'chunk_length': [3, 6, 9]
    # 'chunk_length': [30]
    'chunk_length': [3]
})

# caption
caption_configs_set = OmegaConf.create({
    'modality': ['caption'],
    # 'seconds_per_caption': [2, 4, 8],
    # 'seconds_per_caption': [8],
    'seconds_per_caption': [2],
    # 'frames_per_caption': [1, 2, 4],
    'frames_per_caption': [4],
    # 'num_return_sequences': [5, 10],
    'num_return_sequences': [10],
    # 'temperature': [1.0]
})
from pathlib import Path
# object
ape_configs = OmegaConf.load(Path(__file__).parent / 'ape.yaml').model2config

object_configs_set = OmegaConf.create({
    'modality': ['object'],
    # 'confidence_threshold': [0.2, 0.25, 0.3, 0.35],
    'confidence_threshold': [0.2],
    'text_prompt': [None],
    # 'object_type': ['Dynamic', 'Static'],  # Static代表检测所有在close set里面的object
    # 'model': list(ape_configs.keys()),
    'model': [list(ape_configs.keys())[0]],
    # 'num_frame': [4, 8, 16],
    'num_frame': [16],
    # 'sampling_method': ['uniform', 'keyframe']
    'sampling_method': ['uniform'],
    # 'frame_idx': []
})

# clip
clip_configs_set = OmegaConf.create({
    'modality': ['clip'],
})
clip_configs_set = OmegaConf.merge(frame_config_set, clip_configs_set)

uniform_modality2config_set = {
    Modality.Frame: frame_config_set,
    Modality.OCR: ocr_configs_set,
    Modality.ASR: asr_configs_set,
    Modality.VisionToken: visiontoken_configs_set,
    Modality.Caption: caption_configs_set,
    Modality.Clip:  clip_configs_set,
    Modality.Object: object_configs_set,
}


class PrepConfig:

    @staticmethod
    def generate_config_combinations(config: DictConfig, modality):
        """
        生成所有可能的组合，并将每个组合转换为字典形式。

        参数:
            config (DictConfig): OmegaConf 配置对象。

        返回:
            list: 包含所有组合的字典列表。
        """
        # 获取所有配置项的键和值
        keys = list(config.keys())
        values = [config[key] for key in keys]

        # 生成所有可能的组合
        combinations = list(itertools.product(*values))

        # 将每个组合转换为字典
        config_dicts = [
            {key: combo[i] for i, key in enumerate(keys)}
            for combo in combinations
        ]

        return PrepConfig.filter_invalid_config(config_dicts, modality)

    @staticmethod
    def filter_invalid_config(config_dicts, modality):
        filtered_config_dicts = []
        for config_dict in config_dicts:
            # caption
            if modality == Modality.Caption:
                if config_dict.get('model'):
                    if config_dict['model'] == CaptionModelName.llava and config_dict['frames_per_caption'] != 1:
                        continue
            # other rules
            filtered_config_dicts.append(config_dict)
        return filtered_config_dicts


    @staticmethod
    def get_valid_modality_combinations():
        valid_modality_combinations = []
        for base_modality in base_modalities:
            for i in range(len(optional_modalities) + 1):
                modality_combination = [base_modality]
                if i == 0:
                    valid_modality_combinations.append(modality_combination)
                else:
                    for combination in itertools.combinations(optional_modalities, i):
                        modality_combination = list(combination)
                        modality_combination.append(base_modality)
                        modality_combination = list(set(modality_combination))  # 排除caption+caption这种特殊情况
                        valid_modality_combinations.append(modality_combination)

        return {PrepConfig.get_modstr(item): item for item in
                valid_modality_combinations}
    # valid_modality_combinations: 一个从modality combination name (e.g., 'caption_object')到modality combination (e.g., ['caption', 'object'])的映射
    # modality2config_set: 一个从modality (e.g., 'caption')到 对应modality的configuration set (e.g. {'modality': ['caption'], 'seconds_per_caption': [2], 'frames_per_caption': [4], 'model': ['lavila']})的映射
    # 根据valid_modality_combinations，对modality2config_set进行排列组合
    # e.g. {'caption': {'frames_per_caption': 4, 'modality': 'caption', 'model': 'lavila', 'seconds_per_caption': 2}, 'ocr': {'modality': 'ocr', 'num_frame': 16, 'sampling_method': 'uniform'}}
    @staticmethod
    def get_valid_modality_config(valid_modality_combinations : dict, modality2config_set):
        valid_modality_config = []
        for _, modality_combination in valid_modality_combinations.items():
            if all([modality in modality2config_set.keys() for modality in modality_combination]):
                valid_modality2configs = {
                    modality : PrepConfig.generate_config_combinations(modality2config_set[modality], modality) for modality in modality_combination
                }
                for config_combination in itertools.product(*valid_modality2configs.values()):
                    valid_modality_config.append({
                        modality : config for modality, config in zip(modality_combination, config_combination)
                    })
        return valid_modality_config




    @staticmethod
    def get_preprocessing_result(modality2config : dict, video_name, dataset_name):
        modality2result = {}
        for modality, config in modality2config.items():
            if modality == Modality.Object:
                continue
            else:
                result = pickle.load(open(preprocess_store_path(modality, config, video_name, dataset_name), 'rb'))
                modality2result[modality] = result
        return modality2result

    @staticmethod
    def get_modstr(selected_modalities: list):
        selected_modalities = sorted(selected_modalities, key=lambda modality: modality2priority[modality])
        return '_'.join(selected_modalities)

    @staticmethod
    def caption_default_config():
        return OmegaConf.create({
            'modality': 'caption',
            'seconds_per_caption': 2,
            'frames_per_caption': 4,
            'num_return_sequences': 10,
            'temperature': 1.0
        })

    @staticmethod
    def frame_default_config():
        return OmegaConf.create({
            'modality': 'frame',
            'num_frame': 16,
            'sampling_method': 'uniform'
        })

    @staticmethod
    def object_default_config():
        return OmegaConf.create({
            'modality': 'object',
            'confidence_threshold': 0.3,
            'text_prompt': None,
            'model': 'APE-L_D',
            'num_frame': 16,
            'sampling_method': 'uniform'
        })

    @staticmethod
    def asr_default_config():
        return OmegaConf.create({
            'modality': 'asr',
            'model': 'whisper-large',
            'chunk_length': 3,
        })

    @staticmethod
    def ocr_default_config():
        return OmegaConf.create({
            'modality': 'ocr',
            'num_frame': 16,
            'sampling_method': 'uniform'
        })

    @staticmethod
    def visiontoken_default_config():
        return OmegaConf.create({
            'modality': 'visiontoken',
            'num_frame': 16,
            'sampling_method': 'uniform'
        })

    @staticmethod
    def clip_default_config():
        return OmegaConf.create({
            'modality': 'clip',
            'num_frame': 16,
            'sampling_method': 'uniform'
        })
