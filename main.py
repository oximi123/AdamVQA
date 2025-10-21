import os
import pickle

from ConfigTuner.config_tuner import constraints_func, VLMTuner
from ConfigTuner.knowledgebase import VideoKnowledgeBase
from ConfigTuner.latency_pruner import TokenLatencyModel, KnobSpec, LatencyAwareSpacePruner
from load_dataset import VideoDatasetLoader
from preprocess_constant import VQADataset, PrepConfigSetType, KeyWordExtractorNames, LLMName, Modality
from script_run_inference import inference_parameter
from experiment_setting import *

from util import get_tuning_result_file_path



if __name__ == '__main__':
    inference_parameter.use_existing_retrieve_result = True
    inference_parameter.use_existing_keyword_result = True
    inference_parameter.do_inference = True
    inference_parameter.need_save_inference_result = True
    inference_parameter.need_save_retrieve_result = True
    inference_parameter.need_save_keyword_result = True
    inference_parameter.force_inference = False

    results = {}
    from tqdm import tqdm

    # latency_constraint = None
    latency_constraint = 0.5
    train_set_ratio = 0.5
    # seed = 0
    # seed = 42
    seed = 123
    repeat_save = False
    sampler_name = 'VLMTuner'
    llm_name = LLMName.LlavaVideoQwen7b
    extractor_name = llm_name
    inference_parameter.llm_name = llm_name

    dataset_loader = VideoDatasetLoader()
    dataset_name = VQADataset.MSVD_QA
    prep_config_set_type = PrepConfigSetType.All
    keyword_extractor_name = KeyWordExtractorNames.LLMExtractor
    video_id2questions, dataset_path, question_type, video_id2name = dataset_loader.load_dataset(dataset_name,
                                                                                                 top_k_video=top_k_video)
    modality2config_set = dataset_loader.get_dataset_inference_config_set(dataset_name, type=prep_config_set_type)
    knowledge_base = VideoKnowledgeBase(llm_name, dataset_name, extractor_name, num_video=base_video_num)
    tlm = TokenLatencyModel(a=0.002, b=0.05)
      # seconds

    # Declare some latency-monotone knobs (larger => more tokens).
    mono_specs = [
        KnobSpec(Modality.VisionToken, "num_frame", "increasing"),
        KnobSpec(Modality.Object, "num_frame", "increasing"),
        KnobSpec(Modality.Object, "threshold", "decreasing"),
        KnobSpec(Modality.OCR, "text_threshold", "decreasing"),
    ]

    pruner = LatencyAwareSpacePruner(
        slo_seconds=latency_constraint,
        token_latency_model=tlm,
        monotone_knobs=mono_specs,
        safety_margin=0.0
    )
    for target_video_id in tqdm(list(video_id2questions.keys())):
        file_path, result_key, result_dir = get_tuning_result_file_path(llm_name,'VLMTuner', dataset_name, target_video_id, 0,
                                                                        seed, train_set_ratio,
                                                                        latency_constraint=latency_constraint,
                                                                        extractor_name=extractor_name)
        if os.path.exists(file_path) and not repeat_save:
            print(
                f'{target_video_id} has finished with {sampler_name}, seed {seed}, latency constraint {latency_constraint}s.')
            continue

        print(target_video_id)
        # if target_video_id != 'O_NWtDShLeg_21_25':
        #     continue
        phase1_sampler_name = 'TPESampler'
        phase2_sampler_name = 'NSGAIISampler'
        vlmtuner = VLMTuner(llm_name, dataset_name, video_id2questions, target_video_id, knowledge_base, 42, 1, constraints_func,
                            phase1_sampler_name=phase1_sampler_name, phase2_sampler_name = phase2_sampler_name, pruner = pruner, tlm = tlm)
        args = {
            'change_thresh': 0.03,
            'min_budget_per_iter': 3,
            'lhs_sample_budget': 3,
            'process_time_budget': 1e9,
            'latency_constraint': latency_constraint,
            'keyword_extractor_name': keyword_extractor_name,
            'question_type': question_type,
            'train_set_ratio': train_set_ratio,
            'test_set_ratio': 0.5,
            'video_id2name': video_id2name,
            'modality2config_set': modality2config_set,
            'seed': seed,
            'target_video_id': target_video_id,
            'total_budget': 30,
            'inference_parameter': inference_parameter
        }
        obj, global_q_scores, global_trials = vlmtuner.tune(**args)

        results = {
            'whole_set_accuracies_history': obj.whole_set_accuracies_history,
            'whole_set_e2elatencys_history': obj.whole_set_e2elatencys_history,
            'trail_history': obj.trail_history,
            'train_ids': obj.org_train_ids,
            'test_ids': obj.test_ids,
            'global_q_scores': global_q_scores,
            'global_trials': global_trials,
        }
        if repeat_save:
            num = len(os.listdir(result_dir))
            file_name = file_path.split('.pkl')[0]
            file_path = f'{file_name}_{num}.pkl'
        pickle.dump(results, open(file_path, 'wb'))