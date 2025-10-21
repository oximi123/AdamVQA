from preprocess_constant import KnobType
from script_evaluation import evaluate_for_one_video
from script_run_inference import run_inference_for_one_video
from util import *

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


class Objective(object):
    def __init__(self,llm_name, video_id2questions, process_time_budget, latency_constraint, dataset_name, video_id, modality2config_set: dict,
                 video_id2name, keyword_extractor_name, question_type,inference_parameter, valid_modality_combinations=None,
                 train_set_ratio=1.0, test_set_ratio=None, seed=42, train_ids = None, test_ids = None, tlm = None):

        self.process_time_budget = process_time_budget
        self.latency_constraint = latency_constraint
        self.dataset_name = dataset_name
        self.video_id = video_id
        self.modality2config_set = modality2config_set
        self.valid_modality_combinations = valid_modality_combinations
        self.train_set_ratio = train_set_ratio
        self.test_set_ratio = test_set_ratio
        self.video_id2name = video_id2name
        self.keyword_extractor_name = keyword_extractor_name
        self.question_type = question_type
        self.whole_set_accuracies_history = []
        self.whole_set_e2elatencys_history = []
        self.processing_time_history = []
        self.trail_history = []
        self.train_id_his = []
        self.video_id2questions = video_id2questions
        self.whole_set_ids = [idx for idx in range(len(self.video_id2questions[self.video_id]))]
        self.train_ids, self.test_ids = vqa_train_test_split(len(self.video_id2questions[self.video_id]),
                                                             self.train_set_ratio, self.test_set_ratio)
        self.llm_name = llm_name
        if train_ids is not None:
            assert test_ids is not None
            self.train_ids, self.test_ids = train_ids, test_ids
        self.org_train_ids = self.train_ids #  用于记录partial optimize 之前的train id
        self.seed = seed
        self.tlm = tlm
        self.inference_parameter = inference_parameter

    def evaluate_config(self, valid_modality2config, use_basic_modality=True, record_result=True):
        # if use_basic_modality:
        kwargs = {}
        kwargs['video_id2questions'] = self.video_id2questions
        kwargs['llm_name'] = self.llm_name
        kwargs['dataset_name'] = self.dataset_name
        kwargs['video_id2name'] = self.video_id2name
        kwargs['video_id'] = self.video_id
        kwargs['video_name'] = self.video_id2name[self.video_id]
        kwargs['valid_modality2config'] = [valid_modality2config]  # inference
        kwargs['modality2config'] = valid_modality2config  # evaluate
        kwargs['selected_modality2config'] = [valid_modality2config]  # load result
        kwargs['keyword_extractor_name'] = self.keyword_extractor_name
        kwargs['train_ids'] = self.train_ids
        kwargs['test_ids'] = self.test_ids
        kwargs['question_type'] = self.question_type
        kwargs['inference_parameter'] = self.inference_parameter
        if len(valid_modality2config) > 0:
            run_inference_for_one_video(**kwargs)
            evaluate_for_one_video(**kwargs)
            whole_set_accuracies, whole_set_e2elatencys, processing_time, modalityconfig2processing_time = load_evaluate_result_for_one_video(
                **kwargs)
        else:
            empty_result = [0] * len(self.video_id2questions[self.video_id])
            whole_set_accuracies, whole_set_e2elatencys, processing_time, modalityconfig2processing_time = empty_result, empty_result, 0, empty_result
        if record_result:
            self.whole_set_accuracies_history.append(whole_set_accuracies)
            self.whole_set_e2elatencys_history.append(whole_set_e2elatencys)
            self.processing_time_history.append(modalityconfig2processing_time)
            self.trail_history.append(kwargs)
            self.train_id_his.append(self.train_ids.copy())
        train_set_accuracies = [whole_set_accuracies[idx] for idx in self.train_ids]
        test_set_accuracies = [whole_set_accuracies[idx] for idx in self.test_ids]
        train_set_latencys = [whole_set_e2elatencys[idx] for idx in self.test_ids]
        test_set_latencys = [whole_set_e2elatencys[idx] for idx in self.test_ids]
        return OmegaConf.create({
            'train_set_accuracy': np.mean(train_set_accuracies).item(),
            'test_set_accuracy': np.mean(test_set_accuracies).item(),
            'train_set_latency': np.mean(train_set_latencys).item(),
            'test_set_latency': np.mean(test_set_latencys).item(),
            'whole_set_accuracy': np.mean(whole_set_accuracies).item(),
            'whole_set_latency': np.mean(whole_set_e2elatencys).item(),
            'processing_time': processing_time,
        })

    def __call__(self, trial: optuna.trial.Trial):
        modalities = list(self.modality2config_set.keys())
        modality2use_modality_knob_names = {modality: f'use_{modality}' for modality in modalities}
        valid_modality2config = {}
        use_basic_modality = False
        for modality, use_modality_knob_name in modality2use_modality_knob_names.items():
            if not modality in Modality.MainModalities:
                continue
            use_modality = trial.suggest_categorical(use_modality_knob_name, [True, False])
            # if use_modality:
            if modality in Modality.BasicModalities:
                use_basic_modality = True
            configs = {}
            for config_name, knob_choice in self.modality2config_set[modality].items():
                if config_name == 'modality':
                    continue
                optuna_knob_name = get_optuna_knob_name(modality, config_name)
                knob_type = judge_knob_type(knob_choice)
                if knob_type == KnobType.CategoricalKnob:
                    knob_value = trial.suggest_categorical(optuna_knob_name, knob_choice)
                elif knob_type == KnobType.FloatKnob:
                    knob_value = trial.suggest_float(optuna_knob_name, min(knob_choice), max(knob_choice) + 1e-4,
                                                     step=knob_choice[1] - knob_choice[0])
                elif knob_type == KnobType.IntKnob:
                    knob_value = trial.suggest_int(optuna_knob_name, min(knob_choice), max(knob_choice),
                                                   step=knob_choice[1] - knob_choice[0])
                configs[config_name] = knob_value
            validate_config(modality, configs)
            if use_modality:
                valid_modality2config[modality] = configs

        cfg = valid_modality2config
        trial.set_user_attr("config", cfg)

        # est_tokens = estimate_token_upper_bound(cfg)
        # est_ttft = self.tlm.tokens_to_ttft(est_tokens)
        # trial.report(est_ttft, step=0)
        # if trial.should_prune():
        #     raise optuna.TrialPruned("SLO violation (estimate)")

        evaluate_result = self.evaluate_config(valid_modality2config, use_basic_modality=use_basic_modality)
        self.process_time_budget = self.process_time_budget - evaluate_result.processing_time
        if self.latency_constraint is not None:
            trial.set_user_attr("ttft_constraint",
                                [evaluate_result.train_set_latency - self.latency_constraint])  # todo 换成p95latency

        # if self.process_time_budget < 0: # todo add budget runs out
        #     print('budget runs out!')
        #     raise TrialPruned()
        per_question_scores = {}
        org_trainset_question_scores = {}
        for q_id, score in enumerate(self.whole_set_accuracies_history[-1]):
            if q_id in self.train_ids:
                per_question_scores[q_id] = score
            if q_id in self.org_train_ids:
                org_trainset_question_scores[q_id] = score
        trial.set_user_attr("per_question_scores",per_question_scores)
        trial.set_user_attr("org_trainset_question_scores",org_trainset_question_scores)
        trial.set_user_attr("train_ids",self.train_ids)

        if self.tlm is not None:
            act_ttft = evaluate_result.train_set_latency
            trial.report(act_ttft, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned("SLO violation (actual)")

        return evaluate_result.train_set_accuracy

    def avg_history(self, part, his_type):
        if his_type == 'accuracy':
            his = self.whole_set_accuracies_history
        else:
            his = self.whole_set_e2elatencys_history

        if part == 'train':
            ids = self.train_ids
        elif part == 'test':
            ids = self.test_ids
        else:
            ids = np.arange(0, len(his[0]))
        return [np.mean([h[idx] for idx in ids]) for
                h in his]

    def train_set_accuracy_history(self):
        return [np.mean([accuracies[idx] for idx in self.train_ids]) for
                accuracies in self.whole_set_accuracies_history]

    def train_set_latency_history(self):
        return [np.mean([latencies[idx] for idx in self.train_ids]) for
                latencies in self.whole_set_e2elatencys_history]

    def test_set_accuracy_history(self):
        return [np.mean([accuracies[idx] for idx in self.test_ids]) for
                accuracies in self.whole_set_accuracies_history]

    def whole_set_accuracy_history(self):
        return [np.mean(accuracies) for
                accuracies in self.whole_set_accuracies_history]

    def set_train_ids(self, train_ids):
        self.train_ids = train_ids
        self.test_ids = [id for id in self.whole_set_ids if id not in train_ids]

