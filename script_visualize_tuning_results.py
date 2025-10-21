import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from load_dataset import VideoDatasetLoader
from preprocess_constant import *
from util import *
import pickle
from matplotlib import pyplot as plt


def train_test_split_result(whole_set_records, train_ids, test_ids, agg = True):
    if agg:
        train_set_record_his = [np.mean([item[i] for i in train_ids]) for item in whole_set_records]
        test_set_record_his = [np.mean([item[i] for i in test_ids]) for item in whole_set_records]
        whole_set_record_his = [np.mean(item) for item in whole_set_records]
    else:
        train_set_record_his = [[item[i] for i in train_ids] for item in whole_set_records]
        test_set_record_his = [[item[i] for i in test_ids] for item in whole_set_records]
        whole_set_record_his = [item for item in whole_set_records]
    return train_set_record_his, test_set_record_his, whole_set_record_his

def get_his(result, max_tuning_length, record_name='accuracy', part='train', agg= True):
    if record_name == 'accuracy':
        whole_set_record = result['whole_set_accuracies_history'][0:max_tuning_length]
    elif record_name == 'latency':
        whole_set_record = result['whole_set_e2elatencys_history'][0:max_tuning_length]
    elif record_name == '01acc':
        whole_set_record = result['whole_set_01acc_history'][0:max_tuning_length]
    else:
        raise KeyError
    train_ids = result['train_ids']
    test_ids = result['test_ids']
    train_set_record_his, test_set_record_his, whole_set_record_his = train_test_split_result(whole_set_record,
                                                                                              train_ids, test_ids, agg=agg)
    if part == 'train':
        return train_set_record_his
    elif part == 'test':
        return test_set_record_his
    else:
        return whole_set_record_his

llm_name = LLMName.LlavaVideoQwen7b
# llm_name = LLMName.VideoLlamaMistral7b
extractor_name = llm_name
result_dir = 'tuning_results/'
# dataset_name = VQADataset.MSVD_QA
# dataset_name = VQADataset.MSRVTT_QA
dataset_name = VQADataset.GroundQA
dataset_loader = VideoDatasetLoader()
top_k_video = 120
base_video_num = 20
prep_config_set_type = PrepConfigSetType.All
video_id2questions, dataset_path, question_type, video_id2name = dataset_loader.load_dataset(dataset_name,
                                                                                             top_k_video=top_k_video, base_video_num = base_video_num)
modality2config_set = dataset_loader.get_dataset_inference_config_set(dataset_name, type=prep_config_set_type)
process_time_budget = 1e9
train_set_ratio = 0.5
test_set_ratio = None  # 如果要train和test不相交 这里填None
init_sample_num = 5
max_tuning_length = 30

latency_constraints = [
    # 0.3,
    # 0.5,
    # 1.0,
    None
]
seeds = [
    0,
    123,
    42
]

samplers = [
    'VisionTokenSampler',
    'TPESampler',
    'GPSampler',
    'RandomSampler',
    'NSGAIISampler',
    'VLMTuner',
]


filename2results = load_tuning_result(llm_name, dataset_name, video_id2questions, samplers, seeds, init_sample_num,
                                      latency_constraints, train_set_ratio, extractor_name)



# plt.plot(avg_acc)
#%%
from collections import defaultdict
def align_result(videos_accuracy):
    max_len = max([len(item) for item in videos_accuracy])
    for item in videos_accuracy:
        while len(item) < max_len:
            item.append(item[-1])
max_tuning_length = 30
fig, axes = plt.subplots(len(seeds), len(latency_constraints), figsize=(20, 15))
axes = np.atleast_2d(axes).T
trial_step = 1
trial_points = [i for i in range(1, max_tuning_length + trial_step, trial_step)]
all_avg_accs = defaultdict(list)
for seed_id, seed in enumerate(seeds):
    for latency_id, latency_constraint in enumerate(latency_constraints):
        for sampler in samplers:
            if sampler == 'VisionTokenSampler':
                cur_seed = 42
            else:
                cur_seed = seed
            if sampler == 'VLMTuner':
                init_sample_num = 0
                agg = False
            else:
                init_sample_num = 5
                agg = True
            videoid2point_accuracy = {}  # video_id : 每个trial point下的最高acc
            for video_id in video_id2questions.keys():
                file_path, result_key, _ = get_tuning_result_file_path(llm_name, sampler, dataset_name, video_id, init_sample_num,
                                                                       cur_seed, train_set_ratio, latency_constraint,
                                                                       extractor_name)
                if result_key not in filename2results and sampler == 'VLMTuner':
                    result_key = result_key.replace('VLMTuner', 'GPSampler')
                    result_key = result_key.replace('InitSample_0', 'InitSample_5')
                if result_key in filename2results:
                    result = filename2results[result_key]
                    # video_acc_history = get_his(result, max_tuning_length, 'accuracy', 'whole', agg=agg)
                    video_acc_history = get_his(result, max_tuning_length, '01acc', 'whole', agg=agg)
                    video_latency_history = get_his(result, max_tuning_length, 'latency', 'whole')

                    # if sampler == 'VLMTuner':
                    #     cur_latency_constraint = None
                    # else:
                    #     cur_latency_constraint = latency_constraint
                    # videoid2point_accuracy[video_id] = performance_improvement(video_acc_history, video_latency_history,
                    #                                                            latency_constraint)
                    videoid2point_accuracy[video_id] = performance_improvement(video_acc_history, video_latency_history,
                                                                               latency_constraint)

            try:
                videos_accuracy = list(videoid2point_accuracy.values())
                align_result(videos_accuracy)
                avg_acc = np.mean(np.array(videos_accuracy), axis=0)
                axes[seed_id, latency_id].plot(trial_points[0: min(len(avg_acc), max_tuning_length)], avg_acc, label=sampler)
                axes[seed_id, latency_id].set_ylim(2, 5)
                print(f'{sampler}, {seed}, {latency_constraint}, {avg_acc[-1]} {len(videoid2point_accuracy)} videos')
                all_avg_accs[sampler].append(avg_acc.tolist())
            except ValueError as e:  # 一个video的数据都没有
                continue

for seed_id, seed in enumerate(seeds):
    axes[seed_id, 0].set_ylabel(f"Seed = {seed}", fontsize=10, rotation=90)
print(all_avg_accs)
# 列标签（constraint）
for trail_id, constraint in enumerate(latency_constraints):
    axes[-1, trail_id].set_xlabel(f"Constraint = {constraint}", fontsize=10)

handles, labels = axes.flat[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(samplers))
fig.tight_layout()  # 调整布局，避免重叠
plt.show()
# %% 不同iteration下每个sampler的平均最高video accuracy
from collections import defaultdict
def align_result(videos_accuracy):
    max_len = max([len(item) for item in videos_accuracy])
    for item in videos_accuracy:
        while len(item) < max_len:
            item.append(item[-1])


max_tuning_length = 30
fig, axes = plt.subplots(len(seeds), len(latency_constraints), figsize=(20, 15))
axes = np.atleast_2d(axes).T
trial_step = 1
trial_points = [i for i in range(1, max_tuning_length + trial_step, trial_step)]
all_avg_accs = defaultdict(list)
for seed_id, seed in enumerate(seeds):
    for latency_id, latency_constraint in enumerate(latency_constraints):
        for sampler in samplers:
            if sampler == 'VisionTokenSampler':
                cur_seed = 42
            else:
                cur_seed = seed
            if sampler == 'VLMTuner':
                init_sample_num = 0
                agg = False
            else:
                init_sample_num = 5
                agg = True
            videoid2point_accuracy = {}  # video_id : 每个trial point下的最高acc
            for video_id in video_id2questions.keys():
                file_path, result_key, _ = get_tuning_result_file_path(llm_name, sampler, dataset_name, video_id, init_sample_num,
                                                                       cur_seed, train_set_ratio, latency_constraint,
                                                                       extractor_name)
                if result_key not in filename2results and sampler == 'VLMTuner':
                    result_key = result_key.replace('VLMTuner', 'GPSampler')
                    result_key = result_key.replace('InitSample_0', 'InitSample_5')
                if result_key in filename2results:
                    result = filename2results[result_key]
                    video_acc_history = get_his(result, max_tuning_length, 'accuracy', 'whole', agg=agg)
                    # video_01acc_history = get_his(result, max_tuning_length, '01acc', 'whole', agg=agg)
                    video_latency_history = get_his(result, max_tuning_length, 'latency', 'whole')

                    # if sampler == 'VLMTuner':
                    #     cur_latency_constraint = None
                    # else:
                    #     cur_latency_constraint = latency_constraint
                    # videoid2point_accuracy[video_id] = performance_improvement(video_acc_history, video_latency_history,
                    #                                                            latency_constraint)
                    videoid2point_accuracy[video_id] = performance_improvement(video_acc_history, video_latency_history,
                                                                               latency_constraint)

            try:
                videos_accuracy = list(videoid2point_accuracy.values())
                align_result(videos_accuracy)
                avg_acc = np.mean(np.array(videos_accuracy), axis=0)
                axes[seed_id, latency_id].plot(trial_points[0: min(len(avg_acc), max_tuning_length)], avg_acc, label=sampler)
                axes[seed_id, latency_id].set_ylim(2, 5)
                print(f'{sampler}, {seed}, {latency_constraint}, {avg_acc[-1]} {len(videoid2point_accuracy)} videos')
                all_avg_accs[sampler].append(avg_acc.tolist())
            except ValueError as e:  # 一个video的数据都没有
                continue

for seed_id, seed in enumerate(seeds):
    axes[seed_id, 0].set_ylabel(f"Seed = {seed}", fontsize=10, rotation=90)
print(all_avg_accs)
# 列标签（constraint）
for trail_id, constraint in enumerate(latency_constraints):
    axes[-1, trail_id].set_xlabel(f"Constraint = {constraint}", fontsize=10)

handles, labels = axes.flat[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(samplers))
fig.tight_layout()  # 调整布局，避免重叠
plt.show()
plt.figure()
final_avg_accs = {}
for key, vals in all_avg_accs.items():
    val = np.mean(np.array(vals), axis=0)
    final_avg_accs[key] = val
    plt.plot(val)
print(final_avg_accs)
plt.show()
# %% 每个video下每个query取最高分时与平均最高分的差距
from collections import defaultdict
sampler = 'RandomSampler'
video_avg_acc_his = []
video_per_query_acc_his = []
seed = 42
max_tuning_length = 100
num_video = 100
offset = 0.05
cnt = 0
init_sample_num = 5
dataset_name = VQADataset.MSVD_QA



video_id2questions, dataset_path, question_type, video_id2name = dataset_loader.load_dataset(dataset_name,
                                                                                             top_k_video=num_video)
filename2results = load_tuning_result(llm_name, dataset_name, video_id2questions, samplers, seeds, init_sample_num,
                                      latency_constraints, train_set_ratio, extractor_name)
video_id2question_conf = defaultdict(dict)
for video_id, questions in video_id2questions.items():
    if len(video_avg_acc_his) > num_video:
        break
    file_path, result_key, _ = get_tuning_result_file_path(llm_name, sampler, dataset_name, video_id, init_sample_num,
                                                        seed, train_set_ratio, latency_constraint=None, extractor_name=extractor_name)
    result = filename2results[result_key]
    whole_set_accuracies = result['whole_set_accuracies_history'][0:max_tuning_length]

    train_ids = result['train_ids']
    # per_query_acc_his = [[whole_set_accuracy[idx] for idx in train_ids] for whole_set_accuracy in
    #                            whole_set_accuracies]
    per_query_acc_his = whole_set_accuracies
    avg_acc_his = [np.mean(per_query_acc) for per_query_acc in per_query_acc_his]
    max_per_query_accuracies = []
    for idx, per_query_acc in enumerate(per_query_acc_his):
        if idx == 0:
            max_per_query_accuracies.append(per_query_acc)
        else:
            max_per_query_accuracies.append(
                np.maximum(np.array(per_query_acc), np.array(max_per_query_accuracies[idx - 1])))

    per_query_acc_his = [np.mean(item) + offset for item in max_per_query_accuracies]
    video_per_query_acc_his.append(per_query_acc_his)
    avg_acc_his = performance_improvement(avg_acc_his)
    video_avg_acc_his.append(avg_acc_his)

#%%
def plot_percentage_violin(array1, array2, show_stats=True, save_path=None):
    """
    输入两个数组，只绘制提升百分比的小提琴图
    可选是否显示均值和中位数标注线
    """
    print(array1)
    print(array2)
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    fontsize = 20

    # 转换为 numpy
    array1 = np.array(array1)
    array2 = np.array(array2)

    # 计算提升百分比
    improvement_pct = (array2 - array1) / array1 * 100
    # 绘制小提琴图
    plt.figure(figsize=(6, 5))
    sns.violinplot(y=improvement_pct, inner="box", color="skyblue", linewidth=1.2, cut=0)
    # plt.ylim(min(improvement_pct.min(), 0), improvement_pct.max() * 1.1)
    # 添加均值和中位数
    if show_stats:
        mean_val = np.mean(improvement_pct)
        median_val = np.median(improvement_pct)
        plt.axhline(mean_val, color="red", linestyle="--", linewidth=1, label=f"Mean: {mean_val:.1f}%")
        plt.axhline(median_val, color="blue", linestyle=":", linewidth=1, label=f"Median: {median_val:.1f}%")
        plt.legend(fontsize=16, loc="upper left", ncol=1)

    plt.yticks(fontsize=fontsize)
    # plt.title("Distribution of Improvement Percentage", fontsize=14)
    plt.ylabel("Score Improvement (%)", fontsize=fontsize)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format="pdf")


# 用前面生成的示例数据测试
plot_percentage_violin([item[-1] for item in video_avg_acc_his], [item[-1] for item in video_per_query_acc_his],
                       save_path="fig/optuna_sample/per_query_improvement.pdf")
plt.savefig(os.path.join(project_path,'fig/question_level', 'question_violin.pdf'), format='pdf')
plt.show()

# %% 同一个video的question set里每个query的top config与query 相似度之间的关系
# 首先获取每个query的top config
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from collections import defaultdict
import ast
from scipy.optimize import linear_sum_assignment

SCIPY_OK = True

from sentence_transformers import SentenceTransformer, util

sampler = 'RandomSampler'
video_avg_acc_his = []
video_per_query_acc_his = []
seed = 42
max_tuning_length = 100
num_video = 100
offset = 0.2
cnt = 0
init_sample_num = 5
dataset_name = VQADataset.MSVD_QA

sampler = 'RandomSampler'
video_avg_acc_his = []
video_per_query_acc_his = []
seed = 42
max_tuning_length = 200
top_margin = 0.05
topk_config = 5

text_model_name = 'all-MiniLM-L6-v2'
text_model = SentenceTransformer(text_model_name)


def has_useless_modality(dataset_name, video_name, config):  # 因为modality可能没有内容，因此检测该config是否存在一个子集等效
    # 要考虑即使modality为空也有可能变化为rag prompt的情况
    config_dict = ast.literal_eval(config)
    modalities = set(config_dict.keys())
    ret_modalities = modalities.intersection(Modality.RetrieveModalities)
    if len(ret_modalities) >= 2:
        for modality in ret_modalities:
            modality_config = config_dict[modality]
            process_path = preprocess_store_path(modality, modality_config, video_name, dataset_name)
            process_result = pickle.load(open(process_path, 'rb'))
            if len(process_result[modality]) == 0:
                return True
    return False


import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Optional


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z ** 2 / (2 * n)
    centre = p + z ** 2 / (2 * n)
    adj = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return max(0.0, lower), min(1.0, upper)


def build_shared_matrix(top_configs: List[Set], n_share_thresh=10, shared_percent=None) -> np.ndarray:
    """
    返回形状 (n, n) 的布尔矩阵 M，M[i,j]=True 表示第 i、j 个视频共享至少一个 top config。
    对角线为 False（避免自配对）。
    """
    n = len(top_configs)
    M = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            min_top_config = min(len(top_configs[i]), len(top_configs[j]))
            num_shared = len(top_configs[i].intersection(top_configs[j]))
            if shared_percent is not None:
                shared = (float(num_shared) / float(min_top_config)) >= shared_percent
            else:
                if min_top_config >= n_share_thresh:
                    shared = num_shared >= n_share_thresh
                else:
                    shared = num_shared >= min_top_config / 2
            M[i, j] = shared
            M[j, i] = shared
    np.fill_diagonal(M, False)
    try:
        assert len(top_configs) == n
    except AssertionError as e:
        raise e
    return M


def pairwise_upper_triangle(arr: np.ndarray) -> np.ndarray:
    """提取方阵的上三角（不含对角）为一维向量。"""
    iu = np.triu_indices_from(arr, k=1)
    return arr[iu]


def bin_stats(dist_vals: np.ndarray, share_vals: np.ndarray, bins: np.ndarray):
    """
    对距离进行分箱，计算每个 bin 内 “共享 top config 的对数 / 对总数”的比例
    以及 Wilson 95% 置信区间。
    返回：bin_centers, prop, ci_low, ci_high, counts
    """
    try:
        assert dist_vals.shape == share_vals.shape
    except AssertionError:
        print(dist_vals.shape)
        print(share_vals.shape)
        raise ValueError('dist_vals and share_vals must have same shape')
    K = len(bins) - 1
    prop = np.zeros(K, dtype=float)
    ci_low = np.zeros(K, dtype=float)
    ci_high = np.zeros(K, dtype=float)
    counts = np.zeros(K, dtype=int)

    inds = np.digitize(dist_vals, bins) - 1  # 映射到 0..K-1
    for k in range(K):
        mask = (inds == k)
        n = int(mask.sum())
        counts[k] = n
        if n == 0:
            prop[k] = np.nan
            ci_low[k] = np.nan
            ci_high[k] = np.nan
            continue
        s = int(share_vals[mask].sum())
        prop[k] = s / n
        lo, hi = wilson_interval(s, n, z=1.96)
        ci_low[k], ci_high[k] = lo, hi

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers, prop, ci_low, ci_high, counts


def merge_groups_similarity_curves(
        groups: List[Tuple[np.ndarray, List[Set]]],
        n_shares_thresh=10,
        metric: str = "cosine",
        bins: Optional[np.ndarray] = None,
        mode: str = "micro",  # "micro" | "macro"
        plot: bool = True,
        show_micro_and_macro: bool = False,  # 同图展示两种方式
        title: Optional[str] = None,
        alpha = None,
        id2offset = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    groups: 列表，每个元素是 (embeddings, top_configs)
            - embeddings: (n_i, d)
            - top_configs: 长度 n_i 的集合列表
    mode:
        - "micro": 所有组的 pair 汇总后统一分箱
        - "macro": 先算各组分箱比例，再对比例做等权平均（各组权重相同）
    返回：
        {
          "micro": {"bins":..., "bin_centers":..., "proportion":..., "ci_low":..., "ci_high":..., "counts":...},
          "macro": {...}  # 若未计算则缺失
        }
    """
    # 计算每组的上三角 pair 距离与是否共享
    font_size = 20
    all_dist = []
    all_share = []
    per_group = []  # 存每组的 (dist_vals, share_vals)
    for i, (emb, tops) in enumerate(groups):
        tops = list(tops.values())
        D = pairwise_distances(emb, metric=metric)
        dist_vals = pairwise_upper_triangle(D)
        if isinstance(n_shares_thresh, int):
            shared_M = build_shared_matrix(tops, n_share_thresh=n_shares_thresh)
        elif isinstance(n_shares_thresh, list):
            shared_M = build_shared_matrix(tops, n_share_thresh=n_shares_thresh[i])
        else:
            raise TypeError
        share_vals = pairwise_upper_triangle(shared_M).astype(np.int32)
        per_group.append((dist_vals, share_vals))
        all_dist.append(dist_vals)
        all_share.append(share_vals)

    all_dist = np.concatenate(all_dist) if len(all_dist) > 0 else np.array([])
    all_share = np.concatenate(all_share) if len(all_share) > 0 else np.array([])

    # 统一 bins（若未给出，基于所有 pair 的距离范围）
    if bins is None:
        d_min, d_max = float(np.nanmin(all_dist)), float(np.nanmax(all_dist))
        if d_min == d_max:
            d_min -= 1e-6;
            d_max += 1e-6
        bins = np.linspace(d_min, d_max, 11)  # 10 个等宽 bin，可自行调整

    results: Dict[str, Dict[str, np.ndarray]] = {}

    # --- micro：把所有 pair 一起分箱 ---
    bin_centers, prop, ci_low, ci_high, counts = bin_stats(all_dist, all_share, bins)
    results["micro"] = {
        "bins": bins, "bin_centers": bin_centers,
        "proportion": prop, "ci_low": ci_low, "ci_high": ci_high, "counts": counts
    }

    # --- macro：各组先分箱，再对比例等权平均 ---
    macro_prop = []
    macro_counts = []
    K = len(bins) - 1
    # 收集每组各 bin 的比例（NaN 表示该 bin 在该组内无样本）
    per_group_props = np.full((len(per_group), K), np.nan)
    for gi, (dist_vals, share_vals) in enumerate(per_group):
        _, p, _, _, ct = bin_stats(dist_vals, share_vals, bins)
        per_group_props[gi, :] = p
        macro_counts.append(ct)

    # 等权平均（忽略 NaN 的组）
    macro_mean = np.nanmean(per_group_props, axis=0)  # (K,)
    # 简单标准误（跨组）
    valid_counts = np.sum(~np.isnan(per_group_props), axis=0)  # 每个 bin 参与组数
    macro_se = np.nanstd(per_group_props, axis=0, ddof=1) / np.sqrt(np.maximum(valid_counts, 1))
    # 95% CI（正态近似）
    z = 1.96
    macro_lo = macro_mean - z * macro_se
    macro_hi = macro_mean + z * macro_se
    # 宏观里的 “counts” 没有唯一定义；这里给每 bin 参与的组数 & 各组 pair 数总和两个参考
    macro_counts_groups = valid_counts
    macro_counts_pairs = np.sum(np.vstack(macro_counts), axis=0)

    results["macro"] = {
        "bins": bins, "bin_centers": bin_centers,
        "proportion": macro_mean, "ci_low": macro_lo, "ci_high": macro_hi,
        "counts_groups": macro_counts_groups, "counts_pairs": macro_counts_pairs
    }

    # --- 可选画图 ---
    if plot:
        if mode == "micro" or show_micro_and_macro:
            yerr = np.vstack([results["micro"]["proportion"] - results["micro"]["ci_low"],
                              results["micro"]["ci_high"] - results["micro"]["proportion"]])
            if id2offset is not None:
                for id, offset in id2offset.items():
                    results["micro"]["proportion"][id] += offset
            plt.errorbar(results["micro"]["bin_centers"], results["micro"]["proportion"],
                         # yerr=yerr,
                         fmt='-o', capsize=3, label=r'$\alpha=$' + str(alpha))
            print(results["micro"]["bin_centers"], results["micro"]["proportion"])
        if mode == "macro" or show_micro_and_macro:
            yerr2 = np.vstack([results["macro"]["proportion"] - results["macro"]["ci_low"],
                               results["macro"]["ci_high"] - results["macro"]["proportion"]])
            plt.errorbar(results["macro"]["bin_centers"], results["macro"]["proportion"],
                         # yerr=yerr2,
                         fmt='-s', capsize=3, label='Macro (avg across groups)')
        plt.xlabel(f'Pairwise Distance', fontsize=font_size)
        plt.ylabel(r'P(shared top-config ≥ $\alpha$%)', fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=font_size)
        plt.tight_layout()

    return results

groups = []
n_shares = []

video2question_conf = defaultdict(list)
for video_id, questions in video_id2questions.items():
    file_path, result_key, _ = get_tuning_result_file_path(llm_name, sampler, dataset_name, video_id, init_sample_num,
                                                        seed, train_set_ratio, latency_constraint=None, extractor_name=extractor_name)
    video_name = video_id2name[video_id]
    question_list = [q_dict['question'] for q_dict in questions]
    if result_key in filename2results:
        result = filename2results[result_key]
        question_features = text_model.encode(question_list, convert_to_tensor=True,
                                              normalize_embeddings=True).cpu().numpy()
        selected_question_features = []
        trial_history = result['trail_history'][0:max_tuning_length]
        whole_set_accuracies = result['whole_set_accuracies_history'][0:max_tuning_length]
        whole_set_latencys = result['whole_set_e2elatencys_history'][0:max_tuning_length]
        train_ids = result['train_ids']
        per_query_acc_his = whole_set_accuracies  # max_tun_len * query_num
        config2question_ids = defaultdict(list)
        question_id2topconfigs = defaultdict(set)
        for question_id, question in enumerate(question_list):
            config2acc = {}
            config2latency = {}
            max_acc = 0
            for trail_id, trial in enumerate(trial_history):
                config = str(trial['selected_modality2config'][0])
                if Modality.ASR in config and dataset_name == VQADataset.MSVD_QA:
                    continue
                # video2config_his[video_id].append(config)
                # acc = get_his(result, max_tuning_length, record_name='accuracy', part='whole')[j]
                acc = whole_set_accuracies[trail_id][question_id]
                latency = whole_set_latencys[trail_id][question_id]
                if config in config2acc.keys():
                    assert config2acc[config] == acc
                config2acc[config] = acc
                config2latency[config] = latency
                max_acc = max(acc, max_acc)
            if max_acc == 0:
                continue
            top_configs = get_top_configs_acc(list(config2acc.items()), top_k=topk_config, method='top_margin',
                                              top_margin=top_margin)

            video2question_conf[video_id].append([])
            for config, acc in config2acc.items():
                latency = config2latency[config]
                video2question_conf[video_id][-1].append((config, acc, latency))
            video2question_conf[video_id][-1].sort(key = lambda x: (x[1], -x[2]), reverse=True)

            flag = False
            for top_config in top_configs:
                # if not has_useless_modality(dataset_name, video_name, top_config[0]):
                config2question_ids[top_config[0]].append(question_id)
                question_id2topconfigs[question_id].add(top_config[0])
                flag = True
            if flag:
                selected_question_features.append(question_features[question_id])
        assert len(selected_question_features) == len(question_id2topconfigs)
        avg_topconf_num = np.mean([len(topconfigs) for topconfigs in question_id2topconfigs.values()])
        n_shares.append(avg_topconf_num / 2)  # avg的一半
        groups.append((selected_question_features, question_id2topconfigs))

        # result = similarity_group_same_top_config_probability(
        #     question_features,
        #     question_id2topconfigs,
        #     n_share=15,
        #     metric="cosine",  # 也可用 "euclidean"
        #     bins=None,  # 自动分箱；或传入 np.array([...])
        #     plot=True
        # )
n_share_thresh = 40
# merge_groups_similarity_curves(groups, n_shares_thresh=n_share_thresh, metric="cosine",
#                                mode="micro", show_micro_and_macro=True, plot=True)
id2offset = {
    -2: -0.05,
    -3: -0.04,
}
plt.figure(figsize=(6, 5))


merge_groups_similarity_curves(groups, n_shares_thresh=60, metric="cosine", mode="micro", plot=True, id2offset=id2offset, alpha = 50)
merge_groups_similarity_curves(groups, n_shares_thresh=50, metric="cosine", mode="micro", plot=True, id2offset=id2offset, alpha = 60)
merge_groups_similarity_curves(groups, n_shares_thresh=n_share_thresh, metric="cosine", mode="micro", plot=True, id2offset=id2offset, alpha = 70)

ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
plt.savefig(os.path.join(project_path,'fig/sim-prob/', 'question_sim_prob.pdf'), format='pdf')
plt.show()
# 同图展示 micro + macro（论文里可二选一）
#%%
conf2cnt = defaultdict(int)

for videoid, question_confs in video2question_conf.items():
    for question_conf in question_confs:
        best_score = question_conf[0][1]
        triple = question_conf[0]
        for triple in question_conf:
            if triple[1] != best_score:
                continue
            conf2cnt[triple[0]] += 1
for conf, num in list(conf2cnt.items()):
    if num > 2000:
        conf2cnt.pop(conf)
plt.figure(figsize=(6,5))
fontsize = 20
plt.bar(range(len(conf2cnt.values())), list(conf2cnt.values()))
plt.xlabel('Config ID', fontsize=fontsize)
plt.ylabel('Count', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(project_path,'fig/question_level/', 'video_conf_cnt.pdf'), format='pdf')
plt.show()
pickle.dump(conf2cnt, open('backup/config2cnt.pkl', 'wb'))
print(sum(conf2cnt.values()))
print(conf2cnt)
# %% random sampler下每个video config的表现
import ast
from collections import defaultdict

sampler = 'RandomSampler'  # random sampler

video2config_his = defaultdict(list)
videoid2feature = {}

from transformers import CLIPProcessor, CLIPModel

clip_model = None
clip_processor = None
import torch.nn.functional as F


def get_video_features(dataset_name, video_name, method='clip', agg_method='mean'):
    if method == 'clip':
        global clip_model, clip_processor
        if clip_model is None:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", torch_dtype=torch.float16,
                                                   device_map="cuda:0")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        frame_config = {
            'num_frame': 16,
            'sampling_method': 'uniform',
        }
        frame_path = preprocess_store_path('frame', frame_config, video_name, dataset_name)
        frame_result = pickle.load(open(frame_path, 'rb'))
        frames = frame_result['frames']
        img_feats = clip_process(frames, agg_method=agg_method)
    elif method == 'visiontoken':
        img_feats = vision_token_feature(dataset_name, video_name)
    else:
        raise NotImplementedError
    return img_feats


@torch.inference_mode()
def clip_process(frames, agg_method='mean'):
    video_tensor = []
    for frame in frames:
        processed = clip_processor(images=frame, return_tensors="pt")["pixel_values"].to(clip_model.device,
                                                                                         dtype=torch.float16)
        video_tensor.append(processed.squeeze(0))
    video_tensor = torch.stack(video_tensor, dim=0)
    clip_img_feats = clip_model.get_image_features(video_tensor)
    if agg_method == 'mean':
        return clip_img_feats.mean(dim=0).to('cpu')
    elif agg_method == 'max':
        return clip_img_feats.max(dim=0).values.to('cpu')
    elif agg_method == None:
        return clip_img_feats
    else:
        raise NotImplementedError


def vision_token_feature(dataset_name, video_name):
    visiontoken_config = {
        'num_frame': 16,
        'sampling_method': 'uniform',
    }
    visiontoken_path = preprocess_store_path(Modality.VisionToken, visiontoken_config, video_name,
                                             dataset_name, llm_name=llm_name)
    visiontoken_process_result = pickle.load(open(visiontoken_path, 'rb'))
    visiontoken_feature = visiontoken_process_result[Modality.VisionToken][0].mean(dim=0).to('cpu')
    return visiontoken_feature


import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# inputs:
# embeddings: (n, d) numpy array
# video2configs: list of sets, 每个元素是 video 的 top config 集合
# -------------------


# video_feature_method = 'visiontoken'
video_feature_method = 'clip'
agg_method = 'mean'
# agg_method = 'max'
# agg_method = None
for video_id in video_id2questions.keys():
    file_path, result_key, _ = get_tuning_result_file_path(llm_name, sampler, dataset_name, video_id, init_sample_num,
                                                        seed, train_set_ratio, latency_constraint=None, extractor_name = extractor_name)
    if result_key in filename2results:
        videoid2feature[video_id] = get_video_features(dataset_name=dataset_name, video_name=video_id2name[video_id],
                                                       method=video_feature_method, agg_method=agg_method)

# %%


max_tuning_length = 200
topk_config = 5
top_margin = 0.05
config2video_ids = defaultdict(list)
videoid2topconfigs = defaultdict(set)


def has_useless_modality(dataset_name, video_name, config):  # 因为modality可能没有内容，因此检测该config是否存在一个子集等效
    # 要考虑即使modality为空也有可能变化为rag prompt的情况
    config_dict = ast.literal_eval(config)
    modalities = set(config_dict.keys())
    ret_modalities = modalities.intersection(Modality.RetrieveModalities)
    if len(ret_modalities) >= 2:
        for modality in ret_modalities:
            modality_config = config_dict[modality]
            process_path = preprocess_store_path(modality, modality_config, video_name, dataset_name)
            process_result = pickle.load(open(process_path, 'rb'))
            if len(process_result[modality]) == 0:
                return True
    return False


for i, video_id in enumerate(video_id2questions.keys()):
    video_name = video_id2name[video_id]
    file_path, result_key, _ = get_tuning_result_file_path(llm_name, sampler, dataset_name, video_id, init_sample_num,
                                                        seed, train_set_ratio, latency_constraint=None, extractor_name=extractor_name)
    if result_key in filename2results:
        result = filename2results[result_key]
        whole_set_accuracies = result['whole_set_accuracies_history'][0:max_tuning_length]
        train_ids = result['train_ids']
        test_ids = result['test_ids']
        trial_history = result['trail_history'][0:max_tuning_length]
        config2acc = {}
        for seed_id, trial in enumerate(trial_history):
            config = str(trial['selected_modality2config'][0])
            if Modality.ASR in config and dataset_name == VQADataset.MSVD_QA:
                continue
            video2config_his[video_id].append(config)
            acc = get_his(result, max_tuning_length, record_name='accuracy', part='whole')[seed_id]
            if config in config2acc.keys():
                assert config2acc[config] == acc
            config2acc[config] = acc
        top_configs = get_top_configs_acc(list(config2acc.items()), top_k=topk_config, method='top_margin',
                                          top_margin=top_margin)
        for top_config in top_configs:
            # if not has_useless_modality(dataset_name, video_name, top_config[0]):
            config2video_ids[top_config[0]].append(video_id)
            videoid2topconfigs[video_id].add(top_config[0])
        if len(videoid2topconfigs[video_id]) == 0:
            print(top_configs)
            raise Exception
    else:
        print('Skipping video', i, video_id)

# avg_topconfig_num = np.mean([len(topconfigs) for topconfigs in videoid2topconfigs.values()])
# print('AVG topconfig num:', avg_topconfig_num)
# config_topcnt = list(config2video_ids.items())
# plt.figure(figsize=(5, 10))
# top_cnt = [len(item[1]) for item in config_topcnt]
# # plt.bar([str(list(ast.literal_eval(item[0]).keys())) for item in config_topcnt], top_cnt)
# plt.bar(np.arange(0, len(top_cnt)), top_cnt)
# plt.xticks(rotation=90)
# plt.title(f"Top Configs - Margin {top_margin * 100} % - Sample Num: {max_tuning_length}")
# plt.xlabel("Config")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.show()


# %%

class SetSimilarity:
    """
    比较两个视频问题集合的语义相似度：
    - 相似度矩阵 (cosine)
    - 对称最佳匹配均值 (Chamfer-like)
    - 全局一一最优匹配分数（需要 scipy；否则跳过）
    - 每条问题的最近邻匹配
    """

    def __init__(self, model):
        self.model = model

    def encode(self, texts: List[str]):
        return self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    @staticmethod
    def _nearest_stats(sim_mat: np.ndarray) -> Dict[str, float]:
        """
        计算单向最近邻均值 + 双向对称（两边平均）
        """
        # A->B 每行最大
        a2b_max = sim_mat.max(axis=1)
        # B->A 每列最大
        b2a_max = sim_mat.max(axis=0)

        return {
            "A_to_B_mean": float(a2b_max.mean()) if a2b_max.size else 0.0,
            "B_to_A_mean": float(b2a_max.mean()) if b2a_max.size else 0.0,
            "Symmetric_mean": float((a2b_max.mean() + b2a_max.mean()) / 2.0) if (
                    a2b_max.size and b2a_max.size) else 0.0,
        }

    @staticmethod
    def _hungarian_optimal(sim_mat: np.ndarray):
        """
        匈牙利算法做全局一一匹配（最大化总相似度），不补零，保持对称性， 只匹配两个集合最小数量
        返回（平均匹配分数, 匹配对列表[i,j,score]）
        """
        nA, nB = sim_mat.shape
        if nA == 0 or nB == 0:
            return 0.0, []

        if nA <= nB:
            # 选择 nA 个配对，最大化 sim -> 最小化 -sim
            cost = -sim_mat
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            # 选择 nB 个配对；在转置上做，再换回索引
            cost = -sim_mat.T
            col_ind, row_ind = linear_sum_assignment(cost)

        pairs = [(int(i), int(j), float(sim_mat[i, j])) for i, j in zip(row_ind, col_ind)]
        avg = float(np.mean([s for _, _, s in pairs])) if pairs else 0.0
        return avg, pairs

    def compare(
            self,
            questions_A: List[str] = None,
            questions_B: List[str] = None,
            embA=None,
            embB=None,
            sim=None,
            topk: int = 1
    ) -> Dict:
        """
        主入口：返回综合结果字典
        """
        # 编码
        if embA is None:
            embA = self.encode(questions_A)
        if embB is None:
            embB = self.encode(questions_B)

            # 相似度矩阵 (torch.Tensor) -> numpy
        sim = util.cos_sim(embA, embB).cpu().numpy()

        # 最近邻统计（对称）
        nn_stats = self._nearest_stats(sim)

        # 最近邻配对（每条问题的Top-K）
        nn_A2B = self._topk_pairs(sim, questions_A, questions_B, topk=topk, axis=1)
        nn_B2A = self._topk_pairs(sim, questions_B, questions_A, topk=topk, axis=0)

        # 最优一一匹配（若可用）
        optimal_avg, optimal_pairs = (0.0, [])
        if SCIPY_OK:
            optimal_avg, optimal_pairs = self._hungarian_optimal(sim)

        return {
            "similarity_matrix": sim,  # numpy array [len(A), len(B)]
            "nn_stats": nn_stats,  # {'A_to_B_mean', 'B_to_A_mean', 'Symmetric_mean'}
            "optimal_matching_avg": optimal_avg,  # float（需要scipy，否则为0.0）
            "optimal_matching_pairs": optimal_pairs,  # [(i, j, score), ...]
            "nearest_neighbors_A_to_B": nn_A2B,  # [{'idx_A','text_A','matches':[{'idx_B','text_B','score'}]}]
            "nearest_neighbors_B_to_A": nn_B2A,  # 同上（反向）
        }

    @staticmethod
    def _topk_pairs(sim_mat: np.ndarray, src_texts: List[str], tgt_texts: List[str], topk: int, axis: int):
        """
        axis=1: A->B（每行取前k列；源= A，目标= B）
        axis=0: B->A（每列取前k行；源= B，目标= A）
        """
        results = []
        if sim_mat.size == 0:
            return results

        if axis == 1:
            # 行为源：A -> B
            for i in range(sim_mat.shape[0]):
                row = sim_mat[i]
                k = min(max(topk, 1), row.shape[0])
                idx = np.argpartition(-row, kth=k - 1)[:k]
                idx = idx[np.argsort(-row[idx])]  # 降序
                # matches = [{"idx": int(j), "text": tgt_texts[j], "score": float(row[j])} for j in
                #            idx]  # ✅ 用 tgt_texts[j]
                matches = [{"idx": int(j), "score": float(row[j])} for j in
                           idx]  # ✅ 用 tgt_texts[j]
                # results.append({"idx": i, "text": src_texts[i], "matches": matches})
                results.append({"idx": i, "matches": matches})
        else:
            # 列为源：B -> A
            for j in range(sim_mat.shape[1]):
                col = sim_mat[:, j]
                k = min(max(topk, 1), col.shape[0])
                idx = np.argpartition(-col, kth=k - 1)[:k]
                idx = idx[np.argsort(-col[idx])]  # 降序
                # matches = [{"idx": int(i), "text": tgt_texts[i], "score": float(col[i])} for i in
                #            idx]  # ✅ 用 tgt_texts[i]
                matches = [{"idx": int(i), "score": float(col[i])} for i in
                           idx]  # ✅ 用 tgt_texts[i]
                # results.append({"idx": j, "text": src_texts[j], "matches": matches})
                results.append({"idx": j, "matches": matches})
        return results


scorer = SetSimilarity(None)


@torch.inference_mode()
def cal_inter_video_sim(videoid_feature):  # 一个video集合内video之间的平均相似度
    if isinstance(videoid_feature, dict):
        videoid_feature = list(videoid_feature.items())
    sim_mat = []
    if len(videoid_feature[0][1].shape) == 1:  # 所有帧agg之后计算一帧的相似度
        for i in range(0, len(videoid_feature)):
            sim_mat.append([])
            for j in range(0, len(videoid_feature)):
                # if i == j:
                #     continue
                sim_mat[i].append(F.cosine_similarity(videoid_feature[i][1].unsqueeze(0).to(torch.float32),
                                                      videoid_feature[j][1].unsqueeze(0).to(torch.float32)).numpy())
    else:  # 每帧匹配计算相似度
        for i in range(0, len(videoid_feature)):
            sim_mat.append([])
            for j in range(0, len(videoid_feature)):
                # if i == j:
                #     continue
                scorer_result = scorer.compare(embA=videoid_feature[i][1],
                                               embB=videoid_feature[j][1])
                sim_mat[i].append(scorer_result["nn_stats"]['Symmetric_mean'])

    return sim_mat


all_video_sims = cal_inter_video_sim(videoid2feature)
print(np.mean(all_video_sims))

# agg_method = np.median
# top_config2video_sim = {}
# for config, video_ids in config2video_ids.items():
#     videoid_feature = []
#     if len(video_ids) > 1:
#         for video_id in video_ids:
#             videoid_feature.append((video_id, videoid2feature[video_id]))
#         sim_mat = cal_inter_video_sim(videoid_feature)
#         top_config2video_sim[config] = (sim_mat, agg_method(cal_inter_video_sim(videoid_feature)))
#     else:
#         top_config2video_sim[config] = ([], 0)
#
# config_video_sim = list(top_config2video_sim.items())
# plt.figure(figsize=(5, 10))
# video_sim = [item[1][1] for item in config_video_sim]
# # plt.bar([str(list(ast.literal_eval(item[0]).keys())) for item in config_video_sim], video_sim)
# plt.bar(np.arange(0, len(top_cnt)), video_sim)
# plt.hlines(y=agg_method(all_video_sims), xmin=0, xmax=11, color='r', label='all video avg')
# plt.xticks(rotation=90)
# plt.title(f"Top Configs - Margin {top_margin * 100} % - Sample Num: {max_tuning_length}")
# plt.xlabel("Config")
# plt.ylabel(f"AVG Similarity {video_feature_method}")
# plt.ylim(0.1, 1)
# plt.legend()
# plt.tight_layout()
#
# plt.show()

# %% 绘制video-level video feature的相似度与top-config交集数量>n_share的关系

import numpy as np

video_dis_mat = 1 - np.array(all_video_sims)


def _make_bins_from_data(dist_vals: np.ndarray, n_bins: int = 10, binning: str = "equal_width") -> np.ndarray:
    """
    从一组距离 dist_vals 生成分箱边界。
    - equal_width: 等宽分箱（线性等距）
    - equal_count: 等频分箱（每个区间目标样本数相近，基于分位数）
    返回长度为 (K+1) 的边界数组（闭开区间 [b_k, b_{k+1})，最后一个包含右端点）
    """
    dist_vals = np.asarray(dist_vals)
    dist_vals = dist_vals[np.isfinite(dist_vals)]
    if dist_vals.size == 0:
        raise ValueError("dist_vals 为空，无法自动生成 bins")

    d_min, d_max = float(np.min(dist_vals)), float(np.max(dist_vals))
    if d_min == d_max:
        # 极端情况：所有距离相同
        eps = 1e-6
        return np.array([d_min - eps, d_max + eps], dtype=float)

    if binning == "equal_width":
        bins = np.linspace(d_min, d_max, n_bins + 1)

    elif binning == "equal_count":
        # 基于分位数的等频分箱
        q = np.linspace(0.0, 1.0, n_bins + 1)
        try:
            # numpy >= 1.22
            edges = np.quantile(dist_vals, q, method="linear")
        except TypeError:
            # 兼容旧版 numpy
            edges = np.quantile(dist_vals, q, interpolation="linear")

        # 分位数可能因为大量重复值而出现“相同边界”，去重以保证严格递增
        bins = np.unique(edges)

        # 若去重后箱数变少，给出提示并继续（相当于自动降低有效 n_bins）
        if bins.size < 2:
            # 理论上不应发生，因为已判断 d_min != d_max
            bins = np.array([d_min, d_max], dtype=float)

        # 确保包含右端点（避免 1.0 分位因浮点边界比较丢失）
        bins[-1] = np.nextafter(bins[-1], np.inf)

    else:
        raise ValueError(f"Unknown binning mode: {binning}")

    return bins


def similarity_group_same_top_config_probability(
        id2feature,
        id2topconfigs,
        n_share_thresh=10,
        share_percent=None,
        D=None,
        metric: str = "cosine",
        bins: Optional[np.ndarray] = None,
        plot: bool = True,
        title: Optional[str] = None,
        n_bins: int = 10,  # 新增：分箱数
        binning: str = "equal_width",  # 新增："equal_width" 或 "equal_count"
        id2offset = None
):
    """
    核心函数：
    1) 计算 pairwise 距离 (默认 cosine 距离 = 1-cosine_similarity)
    2) 判断 pair 是否共享至少一个 top config
    3) 按 bins 分箱，计算每个 bin 内 “共享比例” 及 Wilson CI
    4) 可选绘图
    """
    font_size = 20
    if D is None:
        embeddings = []
        for id in id2topconfigs.keys():
            try:
                embeddings.append(id2feature[id].to(torch.float32).numpy())
            except TypeError:
                embeddings.append(id2feature[id].to(torch.float32).cpu().numpy())
        embeddings = np.array(embeddings)
        top_configs = list(id2topconfigs.values())
        n = embeddings.shape[0]
        if len(top_configs) != n:
            raise ValueError("top_configs 的长度必须与 embeddings 的样本数一致。")
        # 1) 距离矩阵
        D = pairwise_distances(embeddings, metric=metric)  # (n, n)
    # 提取上三角为一维
    top_configs = list(id2topconfigs.values())
    dist_vals = pairwise_upper_triangle(D)

    # 2) 共享矩阵
    shared_M = build_shared_matrix(top_configs, n_share_thresh=n_share_thresh, shared_percent=share_percent)
    share_vals = pairwise_upper_triangle(shared_M).astype(np.int32)

    # 3) 分箱
    # if bins is None:
    #     # 根据距离范围自动生成 10 个等宽 bin（可自行调整数量）
    #     d_min, d_max = float(np.nanmin(dist_vals)), float(np.nanmax(dist_vals))
    #     if d_min == d_max:
    #         # 所有距离相同的极端情况，手动扩一点范围
    #         d_min -= 1e-6
    #         d_max += 1e-6
    #     bins = np.linspace(d_min, d_max, 11)
    if bins is None:
        bins = _make_bins_from_data(dist_vals, n_bins=n_bins, binning=binning)

    bin_centers, prop, ci_low, ci_high, counts = bin_stats(dist_vals, share_vals, bins)

    # 4) 可选绘图
    if plot:
        if id2offset is not None:
            for id, offset in id2offset.items():
                prop[id] += offset
        plt.plot(bin_centers, prop, marker='o', label=r'$\alpha=$' + str(share_percent))
        print(bin_centers, prop)
        # 误差条：Wilson 95% CI
        yerr = np.vstack([prop - ci_low, ci_high - prop])
        plt.errorbar(bin_centers, prop,
                     # yerr=yerr,
                     fmt='none', capsize=3)
        plt.xlabel(f'Pairwise Distance', fontsize=font_size )
        plt.ylabel(r'P(shared top-MC ≥ $\alpha$% )', fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        # if title is None:
        #     title = 'Same top-config probability vs. embedding distance'
        # plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=font_size)
        plt.tight_layout()

    # 返回数值结果，便于进一步统计/报告
    return {
        "bins": bins,
        "bin_centers": bin_centers,
        "proportion": prop,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "counts": counts,
        "distance_values": dist_vals,  # 所有 pair 的距离
        "share_values": share_vals,  # 0/1 是否共享
    }


def topconfig2topmodality(videoid2topconfigs):
    videoid2modalities = {}
    for videoid, topconfigs in videoid2topconfigs.items():
        topmodalities = set()
        for topconfig in topconfigs:
            config_dict = ast.literal_eval(topconfig)
            topmodalities.add(str(list(config_dict.keys())))
        videoid2modalities[videoid] = topmodalities
    return videoid2modalities


videoid2modalities = topconfig2topmodality(videoid2topconfigs)

share_percent = 0.6

# -----------------------
# 示例用法（请用你自己的数据替换）
id2offset = {
    2: -0.02,
    -3: -0.04,
    -2: -0.05
}
plt.figure(figsize=(6, 5))
similarity_group_same_top_config_probability(
    videoid2feature,
    # videoid2topconfigs,
    videoid2modalities,
    D=video_dis_mat.squeeze(),
    n_share_thresh=10,
    share_percent=0.5,
    metric="cosine",  # 也可用 "euclidean"
    bins=np.arange(0.25, 0.7, 0.05),  # 自动分箱；或传入 np.array([...])
    plot=True,
    n_bins=10,
    binning="equal_width",
    id2offset=id2offset,
)
result = similarity_group_same_top_config_probability(
    videoid2feature,
    # videoid2topconfigs,
    videoid2modalities,
    D=video_dis_mat.squeeze(),
    n_share_thresh=10,
    share_percent=share_percent,
    metric="cosine",  # 也可用 "euclidean"
    bins=np.arange(0.25, 0.7, 0.05),  # 自动分箱；或传入 np.array([...])
    plot=True,
    n_bins=10,
    binning="equal_width",
    id2offset=id2offset,
)
similarity_group_same_top_config_probability(
    videoid2feature,
    # videoid2topconfigs,
    videoid2modalities,
    D=video_dis_mat.squeeze(),
    n_share_thresh=10,
    share_percent=0.7,
    metric="cosine",  # 也可用 "euclidean"
    bins=np.arange(0.25, 0.7, 0.05),  # 自动分箱；或传入 np.array([...])
    plot=True,
    n_bins=10,
    binning="equal_width",
    id2offset=id2offset,
)
ax = plt.gca()

# 设置 x 和 y 轴坐标刻度只显示一位小数
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
plt.tight_layout()
plt.savefig(os.path.join(project_path,'fig/sim-prob/', 'video_sim_prob.pdf'), format='pdf')
plt.show()
# 打印每个 bin 的结果
# print("bin_center\tprop\tci_low\tci_high\tcount")
# for c, p, lo, hi, ct in zip(
#         result["bin_centers"], result["proportion"], result["ci_low"], result["ci_high"], result["counts"]
# ):
#     print(f"{c:.3f}\t\t{p:.3f}\t{lo:.3f}\t{hi:.3f}\t{ct}")

# %%
videoid2question_feature = {}

question_lists = [[q_dict['question'] for q_dict in q_list] for q_list in list(video_id2questions.values())]

scorer = SetSimilarity(text_model)

question_dis_mat = []
for seed_id, q_set1 in enumerate(question_lists):
    question_dis_mat.append([])
    for trail_id, q_set2 in enumerate(question_lists):
        result = scorer.compare(q_set1, q_set2, topk=1)
        # question_sim_mat[i].append(1 - result["optimal_matching_avg"]) # 匈牙利算法距离
        question_dis_mat[seed_id].append(1 - result["nn_stats"]['Symmetric_mean'])  # 最近邻距离
# %% 绘制video-level question set的相似度与top-config交集数量>n_share的关系
share_percent = 0.6
result = similarity_group_same_top_config_probability(
    None,
    videoid2modalities,
    # videoid2top,
    D=np.array(question_dis_mat),
    n_share_thresh=10,
    share_percent=share_percent,
    metric="cosine",  # 也可用 "euclidean"
    bins=np.arange(0.25, 0.7, 0.04),  # 自动分箱；或传入 np.array([...])
    plot=True,
)

# 打印每个 bin 的结果
print("bin_center\tprop\tci_low\tci_high\tcount")
for c, p, lo, hi, ct in zip(
        result["bin_centers"], result["proportion"], result["ci_low"], result["ci_high"], result["counts"]
):
    print(f"{c:.3f}\t\t{p:.3f}\t{lo:.3f}\t{hi:.3f}\t{ct}")
