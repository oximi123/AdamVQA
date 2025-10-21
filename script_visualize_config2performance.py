import os
from types import NoneType

from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from iterate_util import iterate_run
from load_dataset import VideoDatasetLoader
from preprocess_config import PrepConfig

from matplotlib import pyplot as plt
import numpy as np
import json

from util import *
from preprocess_constant import *
import seaborn as sns
import pandas as pd


# %%


def file_name2vis_name(file_name: str, mode=PrepConfigSetType.Combination):  # 把文件名轉化為可視化的名稱,目前是只考虑modality，之后加入config
    if mode == PrepConfigSetType.Combination:
        modality_list = []
        for modality in Modality.MainModalities:
            if modality in file_name:
                modality_list.append(modality)
        modality_list = sort_modalities(modality_list)
        return '+'.join(modality2abbr(modality_list))
    else:
        vis_name = file_name
        for modality in Modality.MainModalities:
            if modality in file_name:
                vis_name = vis_name.replace(modality, modality2abbr(modality))
        return vis_name


def query_level_best_modality_combination(file2eval_results, file2ttfts, question_type):
    if question_type == QuestionType.OE:
        num_question = len(next(iter(file2eval_results.values())))
    elif question_type == QuestionType.MC:
        num_question = len(next(iter(file2eval_results.values()))['scores'])
    num_combination = len(file2eval_results.keys())
    combinations = list(file2eval_results.keys())
    question_score_mat = np.zeros((num_question, num_combination))
    question_ttft_mat = np.zeros((num_question, num_combination))
    for j, combination in enumerate(combinations):
        for i in range(num_question):
            if question_type == QuestionType.OE:
                question_score_mat[i][j] = list(file2eval_results[combination].values())[i][0]['score']
            else:
                question_score_mat[i][j] = file2eval_results[combination]['scores'][i]
            question_ttft_mat[i][j] = file2ttfts[combination][i]

    max_scores = np.max(question_score_mat, axis=1)
    question2best_combination_idxs = [np.where(question_score_mat[i] == max_scores[i])[0] for i in range(num_question)]
    question2best_combination_idx = []
    for i in range(num_question):
        best_combination_idxs = question2best_combination_idxs[i]

        if len(best_combination_idxs) == 1:
            # 唯一的最佳组合
            question2best_combination_idx.append(best_combination_idxs[0])
        else:
            # 存在多个，选择 ttft 最短的那个组合
            ttfts_of_best = question_ttft_mat[i, best_combination_idxs]
            min_ttft_idx = best_combination_idxs[np.argmin(ttfts_of_best)]
            question2best_combination_idx.append(min_ttft_idx)

    return np.array(question2best_combination_idx)


def plot_best_combination_count(vis_name2combination_count, dataset_name, video_id=None, keyword_extractor_name=''):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    keys = list(vis_name2combination_count.keys())
    values = list(vis_name2combination_count.values())

    # 绘制柱状图
    ax.bar(keys, values)
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel('Count')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', linestyle='--', alpha=.3)
    plt.tight_layout()

    if save_fig:
        fig_dir = f'fig/{dataset_name}/QueryLevel'
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        if video_id is not None:
            plt.savefig(os.path.join(fig_dir, keyword_extractor_name, f'{video_id}_combination_count.pdf'),
                        format='pdf')
        else:
            plt.savefig(os.path.join(fig_dir, keyword_extractor_name, 'all_video_combination_count.pdf'), format='pdf')
    plt.show()


def get_vis_results(llm_name, dataset_path, dataset_name, video_id, selected_modality2config, prep_config_set_type,
                    question_type, include_keyword_extraction_time=False,
                    keyword_extractor_name=''):
    file2eval_results, file2inference_results, file2config = get_file2results(llm_name, dataset_name, video_id,
                                                                              selected_modality2config,
                                                                              question_type,
                                                                              include_keyword_extraction_time=False,
                                                                              keyword_extractor_name=keyword_extractor_name)
    vis_name2performance = {}
    vis_name2ttfts = {}
    file2ttfts = {}
    for file, results in file2eval_results.items():
        if question_type == QuestionType.OE:
            vis_name2performance[file_name2vis_name(file, mode=prep_config_set_type)] = extract_scores(results)
        else:
            vis_name2performance[file_name2vis_name(file, mode=prep_config_set_type)] = extract_accuracy(results)
    for file, results in file2inference_results.items():
        ttfts_e2e, ttfts_llm = extract_ttfts(results, filename2modalities(file), keyword_extractor_name)
        if include_keyword_extraction_time:
            file2ttfts[file] = ttfts_e2e
            vis_name2ttfts[file_name2vis_name(file, mode=prep_config_set_type)] = ttfts_e2e
        else:
            file2ttfts[file] = ttfts_llm
            vis_name2ttfts[file_name2vis_name(file, mode=prep_config_set_type)] = ttfts_llm

    return file2eval_results, file2inference_results, file2ttfts, vis_name2performance, vis_name2ttfts, file2config


def visualize_video_result(llm_name, dataset_path, dataset_name, video_id, selected_modality2config,
                           prep_config_set_type,
                           question_type, require_error_bar=False, include_keyword_extraction_time=True,
                           keyword_extractor_name=''):
    file2eval_results, file2inference_results, file2ttfts, vis_name2performance, vis_name2ttfts = get_vis_results(
        llm_name,
        dataset_path, dataset_name, video_id, selected_modality2config, prep_config_set_type,
        question_type, include_keyword_extraction_time=include_keyword_extraction_time,
        keyword_extractor_name=keyword_extractor_name)

    vis_names = list(vis_name2performance.keys())
    question2best_combination_id = query_level_best_modality_combination(file2eval_results, file2ttfts, question_type)
    vis_name2combination_count = {
        vis_name: np.sum(question2best_combination_id == vis_id) for vis_id, vis_name in enumerate(vis_names)
    }
    # plot_best_combination_count(vis_name2combination_count, dataset_name, video_id, keyword_extractor_name)

    keys = list(vis_name2performance.keys())
    n = len(keys)
    x = np.arange(n) * 2.0  # 每组基准 x
    bar_w = 0.6
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()
    scores_mean = [np.mean(vis_name2performance[k]) for k in keys]
    ttfts_mean = [np.mean(vis_name2ttfts[k]) for k in keys]
    # ① 统计
    if require_error_bar:
        scores_std = [np.std(vis_name2performance[k], ddof=1) for k in keys]  # 1σ 误差棒
        ttfts_std = [np.std(vis_name2ttfts[k], ddof=1) for k in keys]

        # ------ 关键修改：非对称误差棒 ------
        # 对 scores 的误差棒（下限不低于0）
        scores_lower_err = np.minimum(scores_mean, scores_std)  # 下限最大为均值
        scores_upper_err = scores_std

        # 对 TTFT 的误差棒（下限不低于0）
        ttfts_lower_err = np.minimum(ttfts_mean, ttfts_std)  # 下限最大为均值
        ttfts_upper_err = ttfts_std

        # ------ 均值柱 + 误差棒 ------
        b1 = ax1.bar(x - bar_w / 2, scores_mean, bar_w,
                     yerr=[scores_lower_err, scores_upper_err],  # 非对称误差棒
                     capsize=4, label='score',
                     color='C0', alpha=.8)
        b2 = ax2.bar(x + bar_w / 2, ttfts_mean, bar_w,
                     yerr=[ttfts_lower_err, ttfts_upper_err],  # 非对称误差棒
                     capsize=4, label='TTFT',
                     color='C1', alpha=.8)

    else:
        b1 = ax1.bar(x - bar_w / 2, scores_mean, bar_w,
                     capsize=4, label='score',
                     color='C0', alpha=.8)
        b2 = ax2.bar(x + bar_w / 2, ttfts_mean, bar_w,
                     capsize=4, label='TTFT',
                     color='C1', alpha=.8)

    ax1.set_ylabel("Score" if question_type == QuestionType.OE else "Accuracy")
    ax2.set_ylabel("TTFT (s)")
    mid = x
    ax1.set_xticks(mid)
    ax1.set_xticklabels(keys, rotation=45, ha="right")
    ax1.set_title(video_id)
    ax1.grid(axis='y', linestyle='--', alpha=.3)

    # handles = b1 + b2
    # labels = [h.get_label() for h in handles]
    # ax1.legend(handles, labels, loc='upper right')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(0.5, 1.15), zorder=20)

    fig.tight_layout()

    if save_fig:
        fig_dir = os.path.join(f'fig/{dataset_name}/{prep_config_set_type}', keyword_extractor_name)
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(f'{fig_dir}/{video_id}_{prep_config_set_type}.pdf', format='pdf')

    plt.show()

    return vis_name2combination_count


def run_visualize_same_extractor(**run_kwargs):
    keyword_extractor_name = run_kwargs['keyword_extractor_name']
    prep_config_set_type = run_kwargs['prep_config_set_type']
    video_id2questions = run_kwargs['video_id2questions']

    valid_modality2config = run_kwargs['valid_modality2config']
    video_id2questions = run_kwargs['video_id2questions']
    video_id2name = run_kwargs['video_id2name']
    question_type = run_kwargs['question_type']
    dataset_path = run_kwargs['dataset_path']
    dataset_name = run_kwargs['dataset_name']
    llm_name = run_kwargs['llm_name']
    all_vis_name2combination_count = None
    for video_id in video_id2questions.keys():
        video_name = video_id2name[video_id]
        print(f'--------------------------{video_name}--------------------------')
        # if question_type == QuestionType.OE:
        vis_name2combination_count = visualize_video_result(llm_name, dataset_path, dataset_name, video_id,
                                                            valid_modality2config, prep_config_set_type,
                                                            question_type,
                                                            include_keyword_extraction_time=include_keyword_extraction_time,
                                                            keyword_extractor_name=keyword_extractor_name)
        plot_best_combination_count(vis_name2combination_count, dataset_name,
                                    keyword_extractor_name=keyword_extractor_name)
        if all_vis_name2combination_count is None:
            all_vis_name2combination_count = vis_name2combination_count
        else:
            for key in all_vis_name2combination_count.keys():
                all_vis_name2combination_count[key] += vis_name2combination_count[key]
        # else:
        #     vis_name2combination_count = visualize_video_result(dataset_path, dataset_name, video_id,
        #                                                         valid_modality2config, prep_config_set_type,
        #                                                         question_type,
        #                                                         include_keyword_extraction_time=include_keyword_extraction_time,
        #                                                         keyword_extractor_name=keyword_extractor_name)
        #     plot_best_combination_count(vis_name2combination_count, dataset_name, keyword_extractor_name=keyword_extractor_name)

    # if question_type == QuestionType.OE:
    plot_best_combination_count(all_vis_name2combination_count, dataset_name,
                                keyword_extractor_name=keyword_extractor_name)


def run_visualize_different_extractors(**run_kwargs):
    keyword_extractor_names = run_kwargs['keyword_extractor_names']
    prep_config_set_type = run_kwargs['prep_config_set_type']
    video_id2questions = run_kwargs['video_id2questions']

    valid_modality2config = run_kwargs['valid_modality2config']
    video_id2questions = run_kwargs['video_id2questions']
    video_id2name = run_kwargs['video_id2name']
    question_type = run_kwargs['question_type']
    dataset_path = run_kwargs['dataset_path']
    dataset_name = run_kwargs['dataset_name']
    llm_name = run_kwargs['llm_name']

    groups = []
    for video_id in video_id2questions.keys():
        performance_data = []
        ttft_data = []
        for keyword_extractor_name in keyword_extractor_names:
            # if keyword_extractor_name == 'LLM':
            #     keyword_extractor_name = ''
            file2eval_results, file2inference_results, file2ttfts, vis_name2performance, vis_name2ttfts, file2config = get_vis_results(
                llm_name,
                dataset_path, dataset_name, video_id, valid_modality2config, prep_config_set_type,
                question_type, include_keyword_extraction_time=True,
                keyword_extractor_name=keyword_extractor_name)
            groups = list(vis_name2performance.keys())
            print(groups)
            performance_data.append(list(vis_name2performance.values()))
            ttft_data.append(list(vis_name2ttfts.values()))
        vis_selected_ids = [0, 2, 4, 6, 8, 10]

        baselines = keyword_extractor_names

        # shape: (baseline, group)

        performance_data = np.array(performance_data)
        if len(performance_data.shape) > 2:
            performance_data = performance_data.mean(axis=2)
        ttft_data = np.array(ttft_data)
        ttft_data = ttft_data.mean(axis=2)

        fontsize = 18
        print(video_id)
        print(repr(performance_data))
        print(repr(ttft_data))
        if prep_config_set_type == PrepConfigSetType.Combination:
            selected_groups = [groups[i] for i in vis_selected_ids]
            selected_performance_data = performance_data[:, vis_selected_ids]
            selected_ttft_data = ttft_data[:, vis_selected_ids]
            # 创建x轴位置
            x = np.arange(len(selected_groups))  # G1~G5
            bar_width = 0.35

            fig, ax1 = plt.subplots(figsize=(12, 4))
            ax2 = ax1.twinx()

            # 颜色映射

            # 绘制 score 的堆叠柱（左y轴）
            score_max_val = 0
            for group_idx in range(len(selected_groups)):
                # 当前 group 的所有 baseline performance 值
                values = [(selected_performance_data[bi][group_idx], bi) for bi in range(len(baselines))]
                values.sort(key=lambda x: (-x[0], x[1]))  # 升序堆叠，低的在底部

                for val, bi in values:
                    score_max_val = max(score_max_val, val)
                    ax1.bar(x[group_idx] - bar_width / 2, val, width=bar_width,
                            label=f'''{'Score' if question_type == QuestionType.OE else 'Accuracy'}''' if group_idx == 0 else "",
                            color=colors[3], hatch='\\', alpha=0.99)

            # TTFT 堆叠图（右轴）
            # colors.reverse()
            ttft_max_val = 0
            for group_idx in range(len(selected_groups)):
                values = [(selected_ttft_data[bi][group_idx], bi) for bi in range(len(baselines))]
                values.sort(key=lambda x: (-x[0], x[1]))  # 升序堆叠

                for val, bi in values:
                    ttft_max_val = max(ttft_max_val, val)
                    ax2.bar(x[group_idx] + bar_width / 2, val, width=bar_width,
                            label=f'TTFT' if group_idx == 0 else "",
                            color=colors[0], hatch='/', alpha=0.99)

            # 设置x轴
            ax1.set_xticks(x)
            # ax1.set_xticklabels(selected_groups, rotation=45, ha='right')
            ax1.set_xticklabels(selected_groups, fontsize=fontsize)

            # 设置y轴
            ax1.set_ylabel("Score" if question_type == QuestionType.OE else 'Accuracy', color='black',
                           fontsize=fontsize)
            ax1.set_ylim(1.5, score_max_val * 1.2)
            ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax1.tick_params(axis='y', labelsize=fontsize)
            ax2.set_ylabel("TTFT", color='black', fontsize=fontsize)
            ax2.set_ylim(0, ttft_max_val * 1.2)
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax2.tick_params(axis='y', labelsize=fontsize)
            # 图例
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            leg = ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper center', fontsize=fontsize, ncol=2)
            # plt.title(video_id2name[video_id])
            leg.set_zorder(20)
            plt.tight_layout()
        elif prep_config_set_type == PrepConfigSetType.CaptionOnly:
            fixed_config = {
                'model': 'lavila'
            }
            x_config = 'frames_per_caption'
            y_config = 'seconds_per_caption'
            modality = Modality.Caption
            plot_heat_map(fixed_config, file2config, x_config, y_config, performance_data, ttft_data, modality)
        elif prep_config_set_type == PrepConfigSetType.VisiontokenOnly:
            modality = Modality.VisionToken
            fixed_config = {}
            x_config = 'sampling_method'
            y_config = 'num_frame'
            print('file2config:', file2config)
            plot_heat_map(fixed_config, file2config, x_config, y_config, performance_data, ttft_data, modality)
        if save_fig:
            plt.savefig(
                f'{project_path}/fig/{dataset_name}/{prep_config_set_type}/{video_id}_{prep_config_set_type}.pdf',
                format='pdf')
        plt.show()


def plot_heat_map(fixed_config, file2config, x_config, y_config, performance_data, ttft_data, modality):
    selected_ids = []
    max_num_value = 4
    x_values = set()
    y_values = set()
    fontsize = 20
    for i, (file, config) in enumerate(file2config.items()):
        flag = True
        for key, val in fixed_config.items():
            if config[modality][key] != val:
                flag = False
                break
        if flag:
            selected_ids.append(i)
            if len(x_values) < max_num_value:
                x_values.add(config[modality][x_config])
            if len(y_values) < max_num_value:
                y_values.add(config[modality][y_config])
    x_values = sorted(x_values)
    y_values = sorted(y_values)
    performance_mat = np.zeros((len(x_values), len(y_values)))
    ttft_mat = np.zeros((len(x_values), len(y_values)))
    annot_mat = []
    for i, x_value in enumerate(x_values):
        annot_mat.append([])
        for j, y_value in enumerate(y_values):
            for idx, (file, config) in enumerate(file2config.items()):
                if idx in selected_ids:
                    if config[modality][x_config] == x_value and config[modality][y_config] == y_value:
                        performance_mat[i, j] = performance_data[:, idx]
                        ttft_mat[i, j] = ttft_data[:, idx]
                        annot_mat[i].append(f'{performance_mat[i, j]:.2f}\n{ttft_mat[i, j]:.2f}s')
                        break

    plt.figure(figsize=(6, 5))
    # plt.imshow(performance_mat)
    # data = pd.DataFrame(performance_mat,
    #                     index=[f"Y{i}" for i in range(5)],
    #                     columns=[f"X{i}" for i in range(5)])

    ax = sns.heatmap(performance_mat, annot=annot_mat, cmap='coolwarm', square=False, fmt="", linewidths=0.5,
                     linecolor="white", annot_kws={"size": fontsize})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize)
    plt.xlabel(y_config, fontsize=fontsize)
    plt.ylabel(x_config, fontsize=fontsize)
    plt.yticks(np.arange(0.5, len(x_values) + 0.5, 1), list([str(x_value) for x_value in x_values]), fontsize=fontsize)
    plt.xticks(np.arange(0.5, len(y_values) + 0.5, 1), list([str(y_value) for y_value in y_values]), fontsize=fontsize)
    plt.tight_layout()
    # plt.figure(figsize=(6, 6))
    # plt.imshow(ttft_mat)
    # plt.xlabel(x_config)
    # plt.ylabel(y_config)
    # plt.show()


if __name__ == '__main__':
    include_keyword_extraction_time = True
    save_fig = True
    # dataset_names = [
    #     VQADataset.NextQA
    # ]
    # keyword_extractors = [
    #     # KeyWordExtractorNames.LLMExtractor,
    #     # KeyWordExtractorNames.KeyBertExtractor,
    #     # KeyWordExtractorNames.BareExtractor,
    # ]
    # pre_config_set_types = [
    #     PrepConfigSetType.Combination,
    #     PrepConfigSetType.VisiontokenOnly,
    #     PrepConfigSetType.CaptionOnly
    # ]

    # iterate_run(dataset_names, pre_config_set_types, keyword_extractors, run_visualize_same_extractor)
    llm_name = LLMName.LlavaVideoQwen7b
    dataset_names = [
        # VQADataset.NextQA,
        VQADataset.MSVD_QA
    ]
    keyword_extractors = [
        llm_name,
        # KeyWordExtractorNames.KeyBertExtractor,
        # KeyWordExtractorNames.BareExtractor,
        # KeyWordExtractorNames.NonExtractor,
    ]
    pre_config_set_types = [
        # PrepConfigSetType.Combination,
        PrepConfigSetType.VisiontokenOnly,
        # PrepConfigSetType.CaptionOnly
    ]
    iterate_run(llm_name, dataset_names, pre_config_set_types, keyword_extractors, run_visualize_different_extractors,
                iter_extractors=False)
