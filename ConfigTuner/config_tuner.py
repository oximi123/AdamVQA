# %%
from abc import abstractmethod
from collections import defaultdict
import ast

from ConfigTuner.latency_pruner import precap_vt_bounds
from ConfigTuner.objective import Objective
from util.preprocess_constant import VQADataset, Modality

import optuna.samplers
from statistics import mean

from util.util import sampler_factory, lhs_sampling, config2optuna_config


def constraints_func(trial):
    return trial.user_attrs["ttft_constraint"]

class ConfigTuner(object):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def tune(self, config_set, budget):
        pass


def VT_token_estimator(frame_num):
    return frame_num * 12





from __future__ import annotations
from typing import Dict, Any, List, Optional, Sequence
import math
import numpy as np
import optuna

# -------------------------------
# Internal imports (version-adaptive)
# -------------------------------
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator as ParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters as ParzenEstimatorParameters

from optuna.terminator import EMMREvaluator
from optuna.terminator import MedianErrorEvaluator
from optuna.terminator import Terminator

def _as_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray([float(v) for v in values], dtype=float)


def _get_tpe_settings(sampler: optuna.samplers.TPESampler) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}

    gamma = getattr(sampler, "_gamma", None)
    if callable(gamma):
        settings["gamma"] = None
    else:
        settings["gamma"] = float(gamma) if gamma is not None else 0.15  # default

    pe_params = None
    for attr in ["_parzen_estimator_parameters", "_pe_params", "_ParzenEstimatorParameters"]:
        if hasattr(sampler, attr):
            pe_params = getattr(sampler, attr)
            break
    if ParzenEstimatorParameters is not None:
        if isinstance(pe_params, ParzenEstimatorParameters):
            settings["pe_params"] = pe_params
        elif isinstance(pe_params, dict):
            settings["pe_params"] = ParzenEstimatorParameters(**pe_params)
        else:
            settings["pe_params"] = ParzenEstimatorParameters()
    else:
        settings["pe_params"] = None

    return settings


def _fit_lg_for_param(
    xs_all: np.ndarray, ys_all: np.ndarray,
    low: float, high: float,
    gamma: Optional[float],
    pe_params
):
    if ParzenEstimator is None or ParzenEstimatorParameters is None:
        raise RuntimeError("Optuna ParzenEstimator internals not available. Please use a compatible Optuna version.")

    if len(xs_all) < 3:
        return None, None

    order = np.argsort(-ys_all)
    xs_sorted = xs_all[order]
    ys_sorted = ys_all[order]

    if gamma is None:
        n_good = max(1, int(0.15 * len(xs_sorted)))
    else:
        n_good = max(1, int(float(gamma) * len(xs_sorted)))

    xs_good = xs_sorted[:n_good]
    l_pe = ParzenEstimator(_as_float_array(xs_good), low, high, False, pe_params)
    g_pe = ParzenEstimator(_as_float_array(xs_all),  low, high, False, pe_params)
    return l_pe, g_pe


def estimate_mc_values_from_tpe_nested(
    sampler: optuna.samplers.TPESampler,
    study: optuna.study.Study,
    nested_space: Dict[str, Dict[str, List[float]]],
    n_samples_per_mc: int = 30,
    mc_list: Optional[List[str]] = None,
    seed: int = 2025,
) -> Dict[str, float]:
    def mc_key_of(cfg: Dict[str, Any]) -> str:
        active = sorted([m for m, v in cfg.items() if isinstance(v, dict) and len(v) > 0])
        return "+".join(active) if active else "EMPTY"
    gamma, pe_params = _get_tpe_settings(sampler)

    trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and isinstance(t.user_attrs.get("config"), dict)
        and t.value is not None
    ]
    if not trials:
        return {m: 0.0 for m in (mc_list or ["ALL"])}

    bounds = {}
    for mod, knobs in nested_space.items():
        for knob, choices in knobs.items():
            key = f"{mod}.{knob}"
            arr = _as_float_array(choices)
            bounds[key] = (float(np.min(arr)), float(np.max(arr)))

    pe_by_param = {}
    for mod, knobs in nested_space.items():
        for knob in knobs.keys():
            key = f"{mod}.{knob}"
            xs, ys = [], []
            for t in trials:
                cfg = t.user_attrs["config"]
                if mod in cfg and isinstance(cfg[mod], dict) and knob in cfg[mod]:
                    try:
                        xs.append(float(cfg[mod][knob]))
                        ys.append(float(t.value))
                    except Exception:
                        continue
            if xs:
                lo, hi = bounds[key]
                xs_all = np.asarray(xs, dtype=float)
                ys_all = np.asarray(ys, dtype=float)
                order = np.argsort(-ys_all)  # maximize
                xs_sorted = xs_all[order]
                n_good = max(1, int((0.15 if gamma is None else gamma) * len(xs_sorted)))
                xs_good = xs_sorted[:n_good]
                l_pe = ParzenEstimator(_as_float_array(xs_good), lo, hi, False, pe_params)
                g_pe = ParzenEstimator(_as_float_array(xs_all),  lo, hi, False, pe_params)
                pe_by_param[key] = (l_pe, g_pe)

    existing_mcs = sorted({mc_key_of(t.user_attrs["config"]) for t in trials})
    target_mcs = existing_mcs if mc_list is None else mc_list

    rng = np.random.default_rng(seed)
    out: Dict[str, float] = {}
    for mk in target_mcs:
        mods = set(mk.split("+")) if mk and mk != "EMPTY" else set()
        logs: List[float] = []
        for _ in range(n_samples_per_mc):
            total = 0.0
            ok = False
            for mod, knobs in nested_space.items():
                if mod not in mods:
                    continue
                for knob, choices in knobs.items():
                    key = f"{mod}.{knob}"
                    if key not in pe_by_param:
                        continue
                    x = float(rng.choice(choices))
                    l_pe, g_pe = pe_by_param[key]
                    logl = float(l_pe.log_pdf(np.array([x], dtype=float))[0])
                    logg = float(g_pe.log_pdf(np.array([x], dtype=float))[0])
                    total += (logl - logg)
                    ok = True
            if ok and math.isfinite(total):
                logs.append(total)
        out[mk] = float(np.mean(np.exp(logs))) if logs else 0.0
    return out





class VLMTuner(ConfigTuner):

    def __init__(self,
                 llm_name,
                 dataset_name,
                 video_id2questions,
                 target_video_id,
                 knowledge_base,
                 seed,
                 num_trial,
                 constraints_func,
                 n_startup_trials=0,
                 top_margin=0.05,
                 top_k=10,
                 transfer_method='top_margin',
                 phase1_sampler_name='TPESampler',
                 phase2_sampler_name='GPSampler',
                 pruner = None,
                 tlm = None
                 ):
        self.phase1_sampler_name = phase1_sampler_name
        self.phase2_sampler_name = phase2_sampler_name
        self.knowledge_base = knowledge_base
        self.seed = seed
        self.n_startup_trials = n_startup_trials
        self.constraints_func = constraints_func
        self.dataset_name = dataset_name
        self.target_video_id = target_video_id
        self.num_trial = num_trial
        self.transfer_method = transfer_method
        self.top_margin = top_margin
        self.top_k = top_k
        self.pruned_space = []
        self.sim_thresh = 0.5
        self.video_ids2questions = video_id2questions
        self.llm_name = llm_name
        self.pruner = pruner
        self.tlm = tlm

    def get_transfer_mcs(self):
        videoid2knowledge, videoid2sims = self.knowledge_base.video_top_configs(target_video_id=self.target_video_id,
                                                                                top_margin=self.top_margin,
                                                                                top_k=self.top_k,
                                                                                method=self.transfer_method)
        videoid2modalities = defaultdict(list)
        for videoid, knowledge in videoid2knowledge.items():
            modality2accs = defaultdict(list)
            for item in knowledge:
                config_str = item['config']
                config_dict = ast.literal_eval(config_str)
                modality2accs[str(list(config_dict.keys()))].append(item['acc'])
            for modality, accs in modality2accs.items():
                videoid2modalities[videoid].append({
                    'modality_combination': modality,
                    'acc': np.mean(accs),
                })
        return videoid2modalities, videoid2sims

    def get_pruned_mc(self, pruned_space, videoid2mcs):
        pruned_videoid2mcs = defaultdict(list)
        for video_id, acc_mc_dicts in videoid2mcs.items():
            for acc_mc_dict in acc_mc_dicts:
                mc = acc_mc_dict['modality_combination']
                if isinstance(mc, str):
                    mc = ast.literal_eval(mc)
                flag = True
                for modal in mc:
                    if modal not in pruned_space:
                        flag = False
                        break
                if flag:
                    pruned_videoid2mcs[video_id].append(acc_mc_dict)
        return pruned_videoid2mcs

    def __get_pruned_space_by_latency(self, dataset_name, space: dict, latency_constraint):
        space = space.copy()
        if dataset_name == VQADataset.MSVD_QA:
            assert Modality.ASR in space
            space.pop(Modality.ASR)
        if latency_constraint is not None:
            pass
            # todo add latency consideration
        return space

    def __need_init(self, visted_mcs, target_mc):
        target_mc = str(target_mc)
        if len(visted_mcs) == 0:
            return True
        if target_mc in visted_mcs:
            return False
        else:
            visted_mcs = [
                set(ast.literal_eval(mc) for mc in visted_mcs)
            ]
            target_mc = set(ast.literal_eval(target_mc))
            for mc in visted_mcs:
                intersect = target_mc.intersection(mc)
                for item in intersect:
                    target_mc.remove(item)
            if len(target_mc) == 0:
                return False
        return True

    def get_best_trial(self, trials: List[optuna.trial.FrozenTrial]):
        best_idx = 0
        best_val = 0
        worst_val = 100
        worst_idx = 100
        for i, trial in enumerate(trials):
            if trial.value > best_val:
                best_val = trial.value
                best_idx = i
            if trial.value < worst_val:
                worst_val = trial.value
                worst_idx = i
        return best_idx, best_val, worst_idx, worst_val

    def __need_next_mc(self, trials: List[optuna.trial.FrozenTrial]):
        tpe_mc_values = trials[-1].user_attrs['tpe_mc_values']
        current_mc = trials[-1].user_attrs['current_mc']
        global_best_trial_id, global_best_val, _, _ = self.get_best_trial(trials)
        if current_mc != sorted(tpe_mc_values.items(), key=lambda x : x[1], reverse=True)[0][0]:
            return True
        else:
            return False

    def __need_next_phase(self, trials: List[optuna.trial.FrozenTrial], max_trial_budget=9):
        best_idx, _, _, _ = self.get_best_trial(trials)
        if len(trials) - best_idx >= max_trial_budget:
            return True
        return False

    def tune(self, **args):
        process_time_budget = args['process_time_budget']
        latency_constraint = args['latency_constraint']
        keyword_extractor_name = args['keyword_extractor_name']
        question_type = args['question_type']
        train_set_ratio = args['train_set_ratio']
        test_set_ratio = args['test_set_ratio']
        video_id2name = args['video_id2name']
        modality2config_set = args['modality2config_set']
        seed = args['seed']
        target_video_id = args['target_video_id']
        total_budget = args['total_budget']
        inference_parameter = args['inference_parameter']

        self.latency_constraint = latency_constraint
        self.pruner.pre_prune()
        self.phase1_sampler, _ = sampler_factory(self.phase1_sampler_name, self.n_startup_trials, seed,
                                               constraints_func if latency_constraint is not None else None)

        emmr_improvement_evaluator = EMMREvaluator()
        median_error_evaluator = MedianErrorEvaluator(emmr_improvement_evaluator)
        terminator = Terminator(
            improvement_evaluator=emmr_improvement_evaluator,
            error_evaluator=median_error_evaluator,
        )

        # step 1: get transfer knowledge (top modality combi)
        pruned_space = self.__get_pruned_space_by_latency(self.dataset_name, modality2config_set,
                                                          latency_constraint)
        obj = Objective(self.llm_name, self.video_ids2questions, process_time_budget, latency_constraint,
                        self.dataset_name, target_video_id, pruned_space,
                        video_id2name, keyword_extractor_name, question_type, inference_parameter,
                        train_set_ratio=train_set_ratio,
                        test_set_ratio=test_set_ratio, seed=seed, tlm = self.tlm)
        search_space = obj.modality2config_set
        def after_trial_callback(study, trial):
            if trial.state != optuna.trial.TrialState.COMPLETE:
                return
            # 直接用 TPESampler + ParzenEstimator 内核计算每个 MC 的 l/g 均值
            mc_values = estimate_mc_values_from_tpe_nested(
                sampler=study.sampler,  # 就是 TPESampler
                study=study,
                nested_space=search_space,
                n_samples_per_mc=50
            )
            study.set_user_attr("tpe_mc_values", mc_values)
        study = optuna.create_study(
            directions=["maximize"],
            sampler=self.phase1_sampler,
            pruner=self.pruner,
            callbacks=[after_trial_callback]
        )

        precap_vt_bounds(
            study, study.pruner,
            vt_grid={"num_frame": [4, 8, 16, 32, 64, 128]},
            base_cfg=obj.modality2config_set,
            token_estimator=VT_token_estimator,  # 强烈建议传入你真实的估计器
        )




        change_thresh = 0.03
        min_budget_per_iter = 3
        lhs_sample_budget = 3
        next_stage_flag = False

        if self.knowledge_base is not None:
            videoid2mcs, videoid2sims = self.get_transfer_mcs()
            videoid2mcs = self.get_pruned_mc(pruned_space, videoid2mcs)
            mc2score = self.weighted_voting(videoid2mcs, videoid2sims, func='-ln')
            mc2budgets = self.assign_budget(total_budget, mc2score, min_budget=min_budget_per_iter)
            sorted_mc_scores = sorted(mc2score.items(), key=lambda item: item[1], reverse=True)
            visted_mcs = set()
            cur_trial_num = 0
            for mc_idx, (cur_mc, score) in enumerate(sorted_mc_scores):
                mc_start_idx = cur_trial_num
                self.update_budget(mc2budgets, mc2score, sorted_mc_scores, mc_idx, min_budget=min_budget_per_iter)
                assert sum(mc2budgets.values()) + len(study.trials) == total_budget
                current_space = {
                    modal: pruned_space[modal] for modal in ast.literal_eval(cur_mc)
                }
                # step 1.1 LHS-based sampling
                if self.__need_init(visted_mcs, cur_mc):
                    mc2budgets[cur_mc] -= lhs_sample_budget
                    lhs_samples = lhs_sampling(current_space, n_samples=lhs_sample_budget, seed=args['seed'])
                    for sample in lhs_samples.values():
                        optuna_sample = config2optuna_config(sample, modality2config_set)
                        study.enqueue_trial(optuna_sample)
                    study.optimize(obj, n_trials=lhs_sample_budget)
                    cur_trial_num += lhs_sample_budget
                    if self.__need_next_mc(study.trials):
                        continue

                # step 1.2 BO-based optimize
                next_mc_flag = False
                while mc2budgets[cur_mc] > 0:
                    modal_choice = {
                        modal: {} for modal in ast.literal_eval(cur_mc)
                    }
                    optuna_sample = config2optuna_config(modal_choice, modality2config_set)
                    study.enqueue_trial(optuna_sample)
                    study.optimize(obj, n_trials=1)
                    cur_trial_num += 1
                    mc2budgets[cur_mc] -= 1
                    if terminator.should_terminate(study):
                        next_stage_flag = True
                        break
                    if self.__need_next_mc(study.trials):
                        next_mc_flag = True
                        break
                    elif mc2budgets[cur_mc] == 0:
                        recycle_budget_needed = lhs_sample_budget
                        for j in range(len(sorted_mc_scores) - 1, -1, -1):
                            recycle_mc = sorted_mc_scores[j][0]
                            if mc2budgets[recycle_mc] > 0:
                                recycle_budget = min(recycle_budget_needed, mc2budgets[recycle_mc])
                                recycle_budget_needed -= recycle_budget
                                mc2budgets[recycle_mc] -= recycle_budget
                                mc2budgets[cur_mc] += recycle_budget
                                if recycle_budget_needed == 0:
                                    break
                if next_stage_flag:
                    break
                if next_mc_flag:
                    assert sum(mc2budgets.values()) + len(study.trials) == total_budget
                    # self.update_budget(mc2budgets, mc2score, sorted_mc_scores, cur_mc, lhs_sample_budget)
                    continue
        else:
            while total_budget > 0:
                study.optimize(obj, n_trials=1)
                if terminator.should_terminate(study):
                    next_stage_flag = True
                    break
                total_budget -= 1


        # Phase 2: query-level optimize
        remained_total_budget = sum(mc2budgets.values())
        global_q_scores = defaultdict(list)
        global_trials: List[optuna.trial.FrozenTrial] = []
        for t in study.get_trials(deepcopy=True):
            if t.state.is_finished() and not t.user_attrs.get("is_warmstart", False):
                global_trials.append(t)
        for whole_set_acc in obj.whole_set_accuracies_history:
            for q_id, acc in enumerate(whole_set_acc):
                if q_id in obj.train_ids:
                    global_q_scores[q_id].append(acc)
        whole_set_train_ids = obj.train_ids

        previous_subset = None
        previous_subset_score = None
        subset_score_no_improve_cnt = 0
        no_improve_thresh = 2
        whole_set_mode = False
        while remained_total_budget > 0:
            if not whole_set_mode:
                subset, subset_score = self.collect_poor_queries(global_q_scores, whole_set_train_ids)
                if previous_subset is None:
                    previous_subset = subset
                    previous_subset_score = subset_score
                else:
                    if subset == previous_subset and not self.has_improvement(previous_subset_score, subset_score):
                        subset_score_no_improve_cnt += 1
                    else:
                        subset_score_no_improve_cnt = 0
                        previous_subset = subset
                        previous_subset_score = subset_score
                    if subset_score_no_improve_cnt >= no_improve_thresh:
                        whole_set_mode = True
                        continue
                if len(subset) == 0:
                    whole_set_mode = True
                    continue
                # avg_subset_score = sum(subset_score) / len(subset_score)
                require_budget = len(subset) / len(whole_set_train_ids)
                partial_study, extra_inference_required = self.rebuild_study_from_global_history(
                    global_trials=global_trials,
                    sampler_name=self.phase2_sampler_name,
                    subset=subset,
                    sampler_seed=self.seed,
                )
                obj.set_train_ids(subset)
                print('query-level optimize')
                partial_study.optimize(obj, n_trials=min_budget_per_iter)

                best_partial_trial_val = mean(subset_score)
                best_partial_trial = None
                for t in partial_study.get_trials(deepcopy=True):
                    if t.state.is_finished() and not t.user_attrs.get("is_warmstart", False):
                        if t.value > best_partial_trial_val:
                            best_partial_trial = t
                            best_partial_trial_val = t.value
                for t in partial_study.get_trials(deepcopy=True):
                    if t.state.is_finished() and not t.user_attrs.get("is_warmstart", False):
                        global_trials.append(t)
                        if t != best_partial_trial:
                            pqs = t.user_attrs.get('per_question_scores')
                        else:
                            pqs = t.user_attrs.get('org_trainset_question_scores')
                        for q_id in whole_set_train_ids:
                            if q_id in pqs:
                                global_q_scores[q_id].append(acc)
                            else:
                                global_q_scores[q_id].append(None)

                if best_partial_trial is None:
                    remained_total_budget -= (require_budget + float(extra_inference_required) / len(
                        whole_set_train_ids)) * min_budget_per_iter
                else:
                    remained_total_budget -= (require_budget + float(extra_inference_required) / len(
                        whole_set_train_ids)) * (min_budget_per_iter - 1) + 1
            else:
                subset = whole_set_train_ids
                new_study, extra_inference_required = self.rebuild_study_from_global_history(
                    global_trials=global_trials,
                    sampler_name=str(self.phase2_sampler_name),
                    subset=subset,
                    sampler_seed=self.seed,
                )
                new_study.optimize(obj, n_trials=int(remained_total_budget))
                for t in new_study.get_trials(deepcopy=True):
                    if t.state.is_finished() and not t.user_attrs.get("is_warmstart", False):
                        global_trials.append(t)
                        pqs = t.user_attrs.get('org_trainset_question_scores')
                        for q_id, acc in pqs.items():
                            global_q_scores[q_id].append(acc)
                remained_total_budget = 0
            print(f'Remained budget: {remained_total_budget}')

        return obj, global_q_scores, global_trials

    def has_improvement(self, previous_scores, current_scores):
        assert len(previous_scores) == len(current_scores)
        is_worse = [
            previous_scores[i] >= current_scores[i] for i in range(len(previous_scores))
        ]
        if all(is_worse):
            return False
        else:
            return True

    def rebuild_study_from_global_history(
            self,
            global_trials: List[optuna.trial.FrozenTrial],
            subset: List[str],
            sampler_name,
            sampler_seed: int,
            direction: str = "maximize",
            shared_percentage_thresh=0.9,
    ):

        if not subset:
            return None
        S = set(subset)
        extra_inference_required = 0
        warm_trials = []
        for ft in global_trials:
            if not ft.state.is_finished():
                continue
            pqs: Dict[str, float] = ft.user_attrs.get("per_question_scores", {})
            # 只用与子集有交集的问题
            overlap_scores = [pqs[q] for q in S if q in pqs]
            shared_percentage = float(len(overlap_scores)) / float(len(subset))
            if shared_percentage < shared_percentage_thresh:
                continue
            extra_inference_required += (len(subset) - len(overlap_scores))
            new_value = mean(overlap_scores)
            warm_trial = optuna.trial.create_trial(
                params=ft.params,
                distributions=ft.distributions,
                value=new_value,
                state=optuna.trial.TrialState.COMPLETE,
                user_attrs={
                    "is_warmstart": True,
                },
            )
            warm_trials.append(warm_trial)
        sampler_kwargs = {
            'population_size' : len(warm_trials)
        }
        sampler, _ = sampler_factory(sampler_name, 0, sampler_seed,
                                     self.constraints_func if self.latency_constraint is not None else None, **sampler_kwargs)
        study = optuna.create_study(direction=direction, sampler=sampler)
        for warm_trial in warm_trials:
            study.add_trial(warm_trial)
        return study, extra_inference_required

    def collect_poor_queries(self, global_q_scores, whole_set_train_ids, agg_method='max',unchange_thresh = 10):
        q_scores = {}
        hard_question_ids = set()
        for train_id, scores in global_q_scores.items():
            if train_id in whole_set_train_ids:
                no_null_scores = [score for score in scores if score is not None]
                q_scores[train_id] = no_null_scores
                is_hard_questions = False
                for i in range(0, len(q_scores[train_id]) - unchange_thresh):
                    max_score = max(q_scores[train_id][i:i + unchange_thresh])
                    if q_scores[train_id][i] >= max_score:
                        is_hard_questions = True
                if is_hard_questions:
                    hard_question_ids.add(train_id)
        if agg_method == 'max':
            per_q_agg_score = {q: max(scores) for q, scores in q_scores.items() if scores}
        elif agg_method == 'mean':
            per_q_agg_score = {q: mean(scores) for q, scores in q_scores.items() if scores}
        if not per_q_agg_score:
            return []
        global_mean = mean(per_q_agg_score.values())
        subset = [q for q, m in per_q_agg_score.items() if m < global_mean and q not in hard_question_ids]
        subset_score = [m for q, m in per_q_agg_score.items() if m < global_mean and q in subset]
        # 兜底：避免空子集
        return subset, subset_score

    def update_budget(self, mc2budgets, mc2score, sorted_mc_scores, mc_idx, min_budget):
        sorted_mcs = [sorted_mc_score[0] for sorted_mc_score in sorted_mc_scores]
        cur_mc = sorted_mcs[mc_idx]
        remained_mcs = sorted_mcs[mc_idx:]
        remained_budget = sum(mc2budgets.values())
        if mc_idx - 1 >= 0:
            previous_mc = sorted_mcs[mc_idx - 1]
            mc2budgets[previous_mc] = 0
        remained_mc2score = {
            mc: mc2score[mc] for mc in remained_mcs
        }
        remained_mc2budgets = self.assign_budget(remained_budget, remained_mc2score, min_budget)
        assert remained_budget == sum(remained_mc2budgets.values())
        for remained_mc, budget in remained_mc2budgets.items():
            mc2budgets[remained_mc] = budget

    def assign_budget(self, total_budget, config2score, min_budget=3):
        total_score = sum(config2score.values())
        for config in config2score:
            config_score = config2score[config] * 1 / total_score
        config2budgets = {
            config: int(score * total_budget) for config, score in config2score.items()
        }
        remain_budget = total_budget - sum(config2budgets.values())
        sorted_config_scores = sorted(config2score.items(), key=lambda x: x[1], reverse=True)
        for config, score in sorted_config_scores:
            if config2budgets[config] < min_budget:
                remain_budget += config2budgets[config]
                config2budgets[config] = 0
        while remain_budget >= min_budget:
            for config, score in sorted_config_scores:
                if config2budgets[config] == 0:
                    config2budgets[config] += min_budget
                    remain_budget -= min_budget
                    if remain_budget < min_budget:
                        break
        top1_config = sorted_config_scores[0][0]
        config2budgets[top1_config] += remain_budget

        return config2budgets

    def weighted_voting(self, id2configs, id2sims, config_key='modality_combination', func='identical'):
        config2score = defaultdict(int)
        score_sum = 0
        for id in id2configs:
            sum_acc = sum([config_acc_dict['acc'] for config_acc_dict in id2configs[id]])
            for config_acc_dict in id2configs[id]:
                config = config_acc_dict[config_key]
                acc = config_acc_dict['acc']
                normed_acc = acc / sum_acc
                if func == 'identical':
                    config_score = id2sims[id] * normed_acc
                elif func == '-ln':
                    config_score = (-np.log(1 - id2sims[id])) ** 2 * normed_acc
                else:
                    raise NotImplementedError
                config2score[config] += config_score
                score_sum += config_score
        for key in config2score:
            config2score[key] = config2score[key] / score_sum
        print(sum(config2score.values()))
        return config2score
