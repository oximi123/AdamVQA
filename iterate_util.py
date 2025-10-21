import copy

from load_dataset import VideoDatasetLoader
from preprocess_config import PrepConfig
from preprocess_constant import PrepConfigSetType
from util import InferenceParameter


def iterate_run(llm_name, dataset_names, prep_config_set_types, keyword_extractors_names, func, inference_parameter : InferenceParameter = None, top_k_video = 10, iter_extractors = True):
    dataset_loader = VideoDatasetLoader()
    inference_parameter_org = copy.deepcopy(inference_parameter)
    for dataset_name in dataset_names:
        if inference_parameter_org is not None:
            inference_parameter = copy.deepcopy(inference_parameter_org)
        video_id2questions_org, dataset_path, question_type, video_id2name = dataset_loader.load_dataset(dataset_name,
                                                                                                     top_k_video=top_k_video)
        for prep_config_set_type in prep_config_set_types:
            video_id2questions = copy.deepcopy(video_id2questions_org)
            prep_modality2config_set = dataset_loader.get_dataset_prep_config_set(dataset_name, type=prep_config_set_type)
            infer_modality2config_set = dataset_loader.get_dataset_inference_config_set(dataset_name, type=prep_config_set_type)
            valid_modality2config = PrepConfig.get_valid_modality_config(PrepConfig.get_valid_modality_combinations(),
                                                                         modality2config_set=infer_modality2config_set)

            num_keyword_extractor = len(keyword_extractors_names)
            if prep_config_set_type == PrepConfigSetType.VisiontokenOnly or prep_config_set_type == PrepConfigSetType.CaptionOnly:
                num_keyword_extractor = 1  # 对于单个modality的configuration测试，由于不需要retrieve，只跑一个keywrodextractor即可
            if iter_extractors: # 查看不同extractor的performance时不需要对每个extractor都画一遍fig，因此不需要遍历每一个extractor
                for keyword_extractor_name in keyword_extractors_names[0:num_keyword_extractor]:
                    print('Dataset:', dataset_name, 'Config Set Type:', prep_config_set_type, 'Keyword Extractor:',
                          keyword_extractor_name)
                    run_kwargs = {
                        'llm_name': llm_name,
                        'video_id2questions': video_id2questions,
                        'dataset_name': dataset_name,
                        'dataset_path': dataset_path,
                        'question_type': question_type,
                        'video_id2name': video_id2name,
                        'modality2config_set': infer_modality2config_set,
                        'keyword_extractor_name': keyword_extractor_name,
                        'valid_modality2config': valid_modality2config,
                        'prep_config_set_type': prep_config_set_type,
                        'inference_parameter': inference_parameter,
                    }
                    func(**run_kwargs)
            else:
                print('Dataset:', dataset_name, 'Config Set Type:', prep_config_set_type, 'Keyword Extractors:',
                      keyword_extractors_names)
                run_kwargs = {
                    'llm_name': llm_name,
                    'video_id2questions': video_id2questions,
                    'dataset_name': dataset_name,
                    'dataset_path': dataset_path,
                    'question_type': question_type,
                    'video_id2name': video_id2name,
                    'modality2config_set': infer_modality2config_set,
                    'keyword_extractor_names': keyword_extractors_names,
                    'valid_modality2config': valid_modality2config,
                    'prep_config_set_type': prep_config_set_type,
                    'inference_parameter': inference_parameter,
                }
                func(**run_kwargs)