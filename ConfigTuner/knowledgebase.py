import ast
import pickle
from collections import defaultdict
from typing import Optional

import math
import numpy as np
import hdbscan
import torch
import torch.functional as F

from ConfigTuner.config_tuner import VLMTuner, constraints_func
from ConfigTuner.latency_pruner import TokenLatencyModel, KnobSpec, LatencyAwareSpacePruner
from util.load_dataset import VideoDatasetLoader
from util.preprocess_constant import Modality, VQADataset
from script_run_inference import inference_parameter


class VideoKnowledgeBase:
    def __init__(self, llm_name, dataset_name, extractor,
                 num_video=20, video_ids=None, samplers=['RandomSampler'], seeds=[42], latency_constraints=[None]):
        self.llm_name = llm_name
        self.dataset_name = dataset_name
        self.num_video = num_video
        self.samplers = samplers
        self.extractor = extractor
        self.video_ids = video_ids
        if self.video_ids is not None:
            self.num_video = len(self.video_ids)
        self.videos = []
        self.video2trails = defaultdict(list)
        self.seeds = seeds
        self.latency_constraints = latency_constraints
        self.video_id2features = {}
        self.video_id2results = defaultdict(list)

        self.__init_knowledgebase()
        pass

    def __init_knowledgebase(self):
        dataset_loader = VideoDatasetLoader()
        if self.video_ids is None:
            self.video_id2questions, self.dataset_path, self.question_type, self.video_id2name \
                = dataset_loader.load_dataset(self.dataset_name, top_k_video=200)
            self.video_ids = []
            for video_id, _ in self.video_id2questions.items():
                self.video_ids.append(video_id)
        else:
            video_id2questions, self.dataset_path, self.question_type, self.video_id2name \
                = dataset_loader.load_dataset(self.dataset_name)
            self.video_id2questions = {}
            for video_id in self.video_ids:
                self.video_id2questions[video_id] = video_id2questions[video_id]
        self.__load_clip_feature_all()
        self.select_kb_indices_hdbscan()
        self.__tune_knowledge_base_video()
        self.__load_trial_history()

    def __load_trial_history(self):
        tuning_results = load_tuning_result(dataset_name=self.dataset_name, video_id2questions=self.video_id2questions,
                                            samplers=self.samplers, seeds=self.seeds,
                                            latency_constraints=self.latency_constraints, extractor_name=self.extractor)
        video_id2processed_config = defaultdict(set)
        for video_id, questions in self.video_id2questions.items():
            for tuning_result_key in tuning_results.keys():
                if video_id in tuning_result_key:
                    result = tuning_results[tuning_result_key]
                    whole_set_acc_his = result['whole_set_accuracies_history']
                    whole_set_latency_his = result['whole_set_e2elatencys_history']
                    trial_his = result['trail_history']
                    train_ids = result['train_ids']
                    test_ids = result['test_ids']
                    retriever = self.extractor
                    for trial, acc, latency in zip(trial_his, whole_set_acc_his, whole_set_latency_his):
                        assert trial['video_id'] == video_id
                        modality2config = trial['modality2config']
                        config_key = f'{modality2config}_{retriever}'
                        if config_key not in video_id2processed_config[video_id]:
                            self.video_id2results[video_id].append({
                                'acc': acc,
                                'latency': latency,
                                'config': modality2config,
                                'retriever': retriever,
                            })
                            video_id2processed_config[video_id].add(config_key)

    def __load_clip_feature(self, video_id, config={'num_frame': 16,
                                                    'sampling_method': 'uniform', }):
        video_name = self.video_id2name[video_id]
        clip_result_path = preprocess_store_path(Modality.Clip, config, video_name, self.dataset_name)
        clip_result = pickle.load(open(clip_result_path, 'rb'))
        clip_feature = clip_result[Modality.Clip]
        return clip_feature

    def __load_clip_feature_all(self):
        for video_id in self.video_ids:
            clip_feature = self.__load_clip_feature(video_id)
            self.video_id2features[video_id] = clip_feature

    def _cluster_hdbscan(self, X,
                         min_cluster_size = 5,
                         min_samples= None,
                         metric= "euclidean",
                         approx_min_span_tree= True,
                         cluster_selection_epsilon= 0.0,
                         cluster_selection_method= "eom",
                         random_state: Optional[int] = 42):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            approx_min_span_tree=approx_min_span_tree,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
        )
        labels = clusterer.fit_predict(X)
        outlier_scores = getattr(clusterer, "outlier_scores_", None)
        probabilities = getattr(clusterer, "probabilities_", None)
        return {
            "clusterer": clusterer,
            "labels": labels,
            "outlier_scores": outlier_scores,
            "probabilities": probabilities,
        }

    def _cluster_to_indices(self, labels):
        clusters = {}
        for i, lb in enumerate(labels):
            clusters.setdefault(int(lb), []).append(i)
        return clusters

    def _allocate_quota_per_cluster(self, cluster_sizes, K):
        non_noise = {c: s for c, s in cluster_sizes.items() if c != -1}
        total = sum(non_noise.values())
        if total == 0:
            return {-1: K}

        quotas = {c: (s / total) * K for c, s in non_noise.items()}
        floor_q = {c: int(math.floor(q)) for c, q in quotas.items()}
        remain = K - sum(floor_q.values())
        frac = sorted(((c, quotas[c] - floor_q[c]) for c in non_noise), key=lambda x: -x[1])
        i = 0
        while remain > 0 and i < len(frac):
            c = frac[i][0]
            floor_q[c] += 1
            remain -= 1
            i += 1
        floor_q[-1] = 0
        return floor_q


    def _medoids_by_distance(self, X, idx, m: int = 1):
        def _cluster_centroid(X, idx):
            return X[idx].mean(axis=0, keepdims=True)
        if len(idx) <= m:
            return list(idx)
        C = _cluster_centroid(X, idx)  # (1, D)
        # L2 distance to centroid
        d = np.linalg.norm(X[idx] - C, axis=1)
        order = np.argsort(d)
        return [idx[i] for i in order[:m]]

    def select_kb_indices_hdbscan(
            self,
            min_cluster_size= 5,
            min_samples= None,
            metric= "euclidean",
    ):
        K = self.num_video
        video_features = self.video_id2features
        cl = self._cluster_hdbscan(
            video_features,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
        )
        labels = cl["labels"]
        clusters = self._cluster_to_indices(labels)

        cluster_sizes = {c: len(idx) for c, idx in clusters.items()}
        quotas = self._allocate_quota_per_cluster(cluster_sizes, K)

        selected_local = []

        non_noise_clusters = sorted([c for c in clusters if c != -1], key=lambda c: -len(clusters[c]))
        for c in non_noise_clusters:
            q = quotas.get(c, 0)
            if q <= 0:
                continue
            idx = clusters[c]
            picks = self._medoids_by_distance(video_features, idx, m=min(q, len(idx)))
            selected_local.extend(picks)

        if len(selected_local) < K and -1 in clusters:
            need = K - len(selected_local)
            noise_idx = clusters[-1]
            out_scores = cl["outlier_scores"]
            if out_scores is None:
                probs = cl["probabilities"]
                if probs is not None:
                    order = np.argsort(-probs[noise_idx])  # high prob first
                    fill = [noise_idx[i] for i in order[:need]]
                else:
                    Gc = video_features.mean(axis=0, keepdims=True)
                    d = np.linalg.norm(video_features[noise_idx] - Gc, axis=1)
                    order = np.argsort(d)
                    fill = [noise_idx[i] for i in order[:need]]
            else:
                order = np.argsort(out_scores[noise_idx])
                fill = [noise_idx[i] for i in order[:need]]
            selected_local.extend(fill)

        selected_local = selected_local[:K]
        self.video_ids = np.array(self.video_ids)[selected_local]

    def cal_video_sim(self, base_video_id, target_video_id, agg_method='mean'):
        with torch.no_grad():
            base_feature = self.video_id2features[base_video_id]
            target_feature = self.__load_clip_feature(target_video_id)
            if agg_method == 'mean':
                base_feature = base_feature.mean(dim=0).to('cpu')
                target_feature = target_feature.mean(dim=0).to('cpu')
                sim = F.cosine_similarity(base_feature.unsqueeze(0).to(torch.float32),
                                          target_feature.unsqueeze(0).to(torch.float32)).numpy()
            else:
                raise NotImplementedError
            return sim

    def filter_out_repeated_config(self, config, video_name):  # 除去部分含有空modality的config，仅考虑ret modal 数量 > 2的情况
        if isinstance(config, str):
            config_dict = ast.literal_eval(config)
        else:
            config_dict = config
        modalities = set(config_dict.keys())
        ret_modalities = modalities.intersection(Modality.RetrieveModalities)
        modality_mask = {}
        if len(ret_modalities) >= 2:
            for modality in ret_modalities:
                modality_config = config_dict[modality]
                process_path = preprocess_store_path(modality, modality_config, video_name, self.dataset_name, self.llm_name)
                process_result = pickle.load(open(process_path, 'rb'))
                if len(process_result[modality]) == 0:
                    modality_mask[modality] = 1
        if len(modality_mask) > 0:
            filtered_config = {}
            for key, val in config_dict.items():
                if key not in modality_mask.keys():
                    filtered_config[key] = val
            if len(filtered_config) > 0:
                return filtered_config
            else:
                return None
        else:
            return config

    def __get_configstr2avgacc(self, video_result, video_name):
        config2avgacc = {}
        for result in video_result:
            config = result['config']
            avgacc = np.mean(result['acc'])
            config = self.filter_out_repeated_config(config, video_name)
            if config is not None:
                config2avgacc[str(config)] = avgacc
        return config2avgacc

    def video_top_configs(self, target_video_id, top_margin=0.05, top_k=10, method='top_margin'):
        videoid2sims = {}
        videoid2knowledge = defaultdict(list)
        for base_video_id in self.video_ids:
            video_sim = self.cal_video_sim(base_video_id, target_video_id)
            video_name = self.video_id2name[base_video_id]
            videoid2sims[base_video_id] = video_sim.item()
            base_video_result = self.video_id2results[base_video_id]
            configstr2avgacc = self.__get_configstr2avgacc(base_video_result, video_name)
            top_configs_acc = get_top_configs_acc(list(configstr2avgacc.items()), top_k=top_k, method=method,
                                                  top_margin=top_margin)
            for config, acc in top_configs_acc:
                videoid2knowledge[base_video_id].append({
                    'config': config,
                    'acc': acc,
                    'extractor': self.extractor
                })
        return videoid2knowledge, videoid2sims

    def __tune_knowledge_base_video(self):
        from tqdm import tqdm

        # latency_constraint = None
        latency_constraint = 0.5
        train_set_ratio = 0.4
        seed = 42
        repeat_save = False
        sampler_name = 'VLMTuner'
        llm_name = LLMName.LlavaVideoQwen7b
        extractor_name = llm_name
        inference_parameter.llm_name = llm_name

        dataset_loader = VideoDatasetLoader()
        dataset_name = VQADataset.MSVD_QA
        prep_config_set_type = PrepConfigSetType.All
        keyword_extractor_name = KeyWordExtractorNames.LLMExtractor
        video_id2questions, dataset_path, question_type, video_id2name = dataset_loader.load_dataset(self.dataset_name,
                                                                                                     top_k_video=200)
        modality2config_set = dataset_loader.get_dataset_inference_config_set(dataset_name, type=prep_config_set_type)
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
        for target_video_id in tqdm(self.video_ids):
            file_path, result_key, result_dir = get_tuning_result_file_path(llm_name, 'VLMTuner', dataset_name,
                                                                            target_video_id, 0,
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
            vlmtuner = VLMTuner(llm_name, dataset_name, video_id2questions, target_video_id, None, 42, 1,
                                constraints_func,
                                phase1_sampler_name=phase1_sampler_name, phase2_sampler_name=phase2_sampler_name,
                                pruner=pruner, tlm=tlm)
            args = {
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

