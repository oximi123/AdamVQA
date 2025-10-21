from iterate_util import iterate_run

import os
from collections import defaultdict
from multiprocessing import Pool

from load_dataset import VideoDatasetLoader
from preprocess_config import PrepConfig
import argparse
from tqdm import tqdm
import json
import openai
import ast
from time import sleep
from preprocess_constant import QuestionType, Modality, VQADataset, PrepConfigSetType, KeyWordExtractorNames
from util import *
import shutil

class ScoreMeter:
    def __init__(self):
        self.score_sum = 0
        self.count = 0
        self.yes_count = 0
        self.no_count = 0
        self.score_dict = {'yes': defaultdict(int), 'no': defaultdict(int)}

    def add_score(self, score, pred):
        self.score_sum += score
        self.count += 1
        pred_lower = pred.lower()
        if 'yes' in pred_lower:
            self.yes_count += 1
            self.score_dict['yes'][score] += 1
        elif 'no' in pred_lower:
            self.no_count += 1
            self.score_dict['no'][score] += 1

    def get_average_score(self):
        res = (self.score_sum / self.count) if self.count else 0
        return f"{res:.6f}"

    def get_accuracy(self, response_type):
        if response_type == 'yes':
            res = (self.yes_count / self.count) if self.count else 0
        elif response_type == 'no':
            res = (self.no_count / self.count) if self.count else 0
        else:
            res = 0
        return f"{res:.6f}"


def annotate(prediction_set, caption_files, output_dir):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    for file in tqdm(caption_files):
        key = file[:-5]  # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred']
        try:
            # Compute the correctness score
            completion = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                model="vicuna-13b-v1.5",
                # model="gpt-3.5-turbo-1106",
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the correctness of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                    }
                ],
                temperature=0.000,
                do_sample=False,
            )
            # Convert response to a Python dictionary.
            response_message = completion["choices"][0]["message"]["content"]
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, qa_set]

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)
            sleep(0.5)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")
            sleep(1)


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    # parser.add_argument("--pred_path", type=str, default=base_path)
    # parser.add_argument("--output_dir", default=os.path.join(base_path, 'results'))
    # parser.add_argument("--output_json", default=os.path.join(base_path, 'results.json'))
    parser.add_argument("--num_tasks", default=16, type=int, help="Number of splits.")
    parser.add_argument("--num_chunks", default=1, type=int, help="Result splits")
    parser.add_argument("--api_version", default=None, type=str, help="OpenAI API version")
    parser.add_argument("--api_key", type=str, default='sk-vSETsflVOMIt0GfByOBAbzkePvOtfxSgJgP9jTrjmLkA7NZM')
    # parser.add_argument("--api_base", type=str, default='https://xiaoai.plus/v1')
    parser.add_argument("--api_base", type=str, default='http://localhost:8000/v1')
    parser.add_argument("--api_type", type=str, default='open_ai')
    args = parser.parse_args()
    return args


def evaluate_OE(pred_contents, output_dir, args, force_overwrite = False, verbose = True):
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.

    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        video_id = sample['video_id']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['id'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)

    # Generating list of id's and corresponding files
    id_list = [x['id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]


    # Generate output directory if not exists.
    if force_overwrite:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['id']
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred']
        qa_set = {"q": question, "a": answer, "pred": pred,
                  "a_type": sample['answer_type'] if 'answer_type' in sample else None}
        prediction_set[id] = qa_set

    # Set the OpenAI API key.
    openai.api_key = args.api_key  # Your API key here
    if args.api_type:
        openai.api_type = args.api_type
    if args.api_version:
        openai.api_version = args.api_version
    if args.api_base:
        openai.api_base = args.api_base  # Your API base here
    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    incomplete_lengths = []
    for _ in range(100):
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"completed_files: {len(completed_files)}")
            print(f"incomplete_files: {len(incomplete_files)}")
            incomplete_lengths.append(len(incomplete_files))
            if len(incomplete_lengths) > 5 and len(set(incomplete_lengths[-5:])) <= 1:
                print(f"incomplete_lengths: {incomplete_lengths}")
                print(f"incomplete_files: {incomplete_files}")
                print(f"completed_files: {completed_files}")
                print(f"failed for 5 times, break")
                break

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, output_dir) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")

    # Combine all the processed files into one
    result_file_name = 'eval_result.json'
    combined_contents = {}
    json_path = os.path.join(output_dir, result_file_name)

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json") and file_name != json_path and file_name != result_file_name:
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                assert 'pred' in content[0], f"Error: {file_name} don't has key=pred"
                assert 'score' in content[0], f"Error: {file_name} don't has key=score"
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    meter_dic = {'total': ScoreMeter()}
    for key, result in combined_contents.items():
        # Computing score
        score_match = result[0]['score']
        score = int(score_match)
        pred = result[0]['pred']

        meter_dic["total"].add_score(score, pred)
        if 'a_type' in result[1] and result[1]['a_type'] is not None:
            typ = str(result[1]['a_type'])
            if typ not in meter_dic:
                meter_dic[typ] = ScoreMeter()
            meter_dic[typ].add_score(score, pred)

            if 'next' in output_dir:
                typ = typ[0]
                if typ not in meter_dic:
                    meter_dic[typ] = ScoreMeter()
                meter_dic[typ].add_score(score, pred)

    csv_dic = {'acc': meter_dic["total"].get_accuracy('yes'), 'score': meter_dic["total"].get_average_score()}

    output = ""
    output += "Yes count: " + str(meter_dic["total"].yes_count) + "\n"
    output += "No count: " + str(meter_dic["total"].no_count) + "\n"
    output += "Accuracy: " + str(meter_dic["total"].get_accuracy('yes')) + "\n"
    output += "Average score: " + str(meter_dic["total"].get_average_score()) + "\n"
    output += "\n"
    output += "Total Score Yes/No distribution:\n"
    for key, value in meter_dic["total"].score_dict.items():
        output += f"{key}:\n"
        for k in range(0, 6):
            v = value[k]
            output += f"{k}: {v}\n"
    output += "\n"
    output += "Answer Type Score distribution:\n"
    output += 'Type, Accuracy, Avg_score\n'
    key_list = sorted([k for k in meter_dic.keys()])
    for key in key_list:
        output += f"{key}, {meter_dic[key].get_accuracy('yes')}, {meter_dic[key].get_average_score()}\n"
        csv_dic[key] = meter_dic[key].get_accuracy('yes')

    output += "\n"
    for k in csv_dic.keys():
        output += f"{k}, "
    output = output.rstrip(', ')  # Remove the trailing comma and space
    output += "\n"

    for k in csv_dic.keys():
        output += str(csv_dic[k]) + ", "
    output = output.rstrip(', ')  # Remove the trailing comma and space
    output += "\n"

    # if verbose:
    #     print(output)
    output_json = json_path
    args.output_csv = output_json.replace(".json", ".csv")
    with open(args.output_csv, 'w') as f:
        f.write(output)
    return meter_dic["total"].get_average_score()

next_qa_answer_map = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
}
def answer_map_MC(dataset_name, answer):
    if dataset_name == VQADataset.NextQA or VQADataset.NextQA:
        return next_qa_answer_map[answer]
    elif dataset_name == VQADataset.ALLVB:
        return answer
    else:
        raise ValueError('Dataset name not supported.')


def evaluate_MC(pred_content, output_dir,dataset_name, modality2config, num_choice=5):
    os.makedirs(output_dir, exist_ok=True)
    num_correct = 0.0
    avg_pure_ttft = 0.0
    avg_whole_ttft = 0.0
    scores = []
    for sample in pred_content:
        pred = sample['pred']
        pred = first_char_as_answer_raw(pred, num_choice=num_choice)
        if pred == answer_map_MC(dataset_name, sample['answer']):
            scores.append(1)
            sample['score'] = 1
            num_correct += 1
        else:
            scores.append(0)
            sample['score'] = 0
        avg_pure_ttft += sample['ttft']
        avg_whole_ttft += get_whole_ttft(sample, modality2config)
    acc = num_correct / len(pred_content)
    avg_pure_ttft = avg_pure_ttft / len(pred_content)
    avg_whole_ttft = avg_whole_ttft / len(pred_content)
    result = {
        'acc': acc,
        'avg_pure_ttft': avg_pure_ttft,
        'avg_whole_ttft': avg_whole_ttft,
        'scores': scores
    }
    json_path = os.path.join(output_dir, "eval_result.json")
    json.dump(result, open(json_path, 'w'))
    return acc, avg_pure_ttft, avg_whole_ttft


def get_ttft(dataset, question, modality2config: dict, modality2result, inference_result, llm_retrieve_result):
    llm_retrieve_time = llm_retrieve_result['retrieve_time']
    modality2time = {}
    for modality in Modality.MainModalities:
        if modality in modality2config.keys():
            modality2time[modality] = modality2result[modality]['retrieve_time']
    llm_ttft = inference_result[question]['']
    return {
        'llm retrieve time': llm_retrieve_time,  # 使用llm获取retrieve result的latency
        **modality2time,
        'llm ttft': llm_ttft,  # 使用llm
    }
    pass

def get_processing_time():
    pass
def get_whole_ttft(sample, modality2config):
    whole_ttft = sample['ttft']
    if len(modality2config.keys()) > 1: # todo
        whole_ttft += sample['llm_retrieve_time'] if sample.get('llm_retrieve_time') else sample['keyword_extraction_time']
    whole_ttft += sample['asr_retrieve_time'] if sample['asr_retrieve_time'] else 0
    whole_ttft += sample['ocr_retrieve_time'] if sample['ocr_retrieve_time'] else 0
    whole_ttft += sample['visiontoken_load_time'] if sample['visiontoken_load_time'] else 0
    # whole_ttft += sample['object_process_time'] if sample['object_process_time'] else 0
    return whole_ttft

def split_train_test(pred_content, test_ratio):
    pass


def evaluate_for_one_video(**run_kwargs):
    video_id2name = run_kwargs['video_id2name']
    dataset_name = run_kwargs['dataset_name']
    modality2config = run_kwargs['modality2config']
    question_type = run_kwargs['question_type']
    keyword_extractor_name = run_kwargs['keyword_extractor_name']
    llm_name = run_kwargs['llm_name']
    video_id = run_kwargs['video_id']
    video_name = video_id2name[video_id]
    train_set_ratio = 0.0 if not run_kwargs.get('train_set_ratio') else run_kwargs['train_set_ratio']
    assert 0 <= train_set_ratio <= 1.0
    print(f'--------------------------{video_name}--------------------------')
    pre_content_name = get_inference_result_dir(llm_name, dataset_name, video_id, modality2config,
                                                keyword_extractor_name=keyword_extractor_name)
    pred_content = json.load(open(pre_content_name, 'r', encoding='utf-8'))
    output_dir = '_'.join([get_inference_result_dir(llm_name, dataset_name, video_id, modality2config,
                                                    keyword_extractor_name=keyword_extractor_name), 'eval_result'])
    if question_type == QuestionType.OE:
        print(f'Config:{modality2config}')
        avg_score = evaluate_OE(pred_content, output_dir, eval_OE_args)
        print(f'Score:{avg_score}')
    elif question_type == QuestionType.MC:
        acc, avg_pure_ttft, avg_whole_ttft = evaluate_MC(pred_content, output_dir, dataset_name, modality2config)
        json.dump(pred_content, open(pre_content_name, 'w', encoding='utf-8'))
        print(f'Config:{modality2config}')
        print(f'Accuracy:{acc}, Pure TTFT:{avg_pure_ttft}, Whole TTFT:{avg_whole_ttft}')
    else:
        raise NotImplementedError


def evaluate_dataset(**run_kwargs):
    video_id2questions = run_kwargs['video_id2questions']
    video_id2name = run_kwargs['video_id2name']
    valid_modality2config = run_kwargs['valid_modality2config']
    for video_id in video_id2questions.keys():
        for modality2config in valid_modality2config:
            run_kwargs['video_id'] = video_id
            run_kwargs['modality2config'] = modality2config
            evaluate_for_one_video(**run_kwargs)
        # video_name = video_id2name[video_id]
        # print(f'--------------------------{video_name}--------------------------')
        # for modality2config in valid_modality2config:
        #     pre_content_name = get_inference_result_dir(dataset_name, video_id, modality2config, keyword_extractor_name=keyword_extractor_name)
        #     pred_content = json.load(open(pre_content_name, 'r', encoding='utf-8'))
        #     output_dir = '_'.join([get_inference_result_dir(dataset_name, video_id, modality2config, keyword_extractor_name=keyword_extractor_name), 'eval_result'])
        #     if question_type == QuestionType.OE:
        #         print(f'Config:{modality2config}')
        #         score = evaluate_OE(pred_content, output_dir, eval_OE_args)
        #         print(f'Score:{score}')
        #     elif question_type == QuestionType.MC:
        #         acc, avg_pure_ttft, avg_whole_ttft = evaluate_MC(pred_content, output_dir, dataset_name, modality2config)
        #         print(f'Config:{modality2config}')
        #         print(f'Accuracy:{acc}, Pure TTFT:{avg_pure_ttft}, Whole TTFT:{avg_whole_ttft}')
        #     else:
        #         raise NotImplementedError
eval_OE_args = parse_args()

if __name__ == '__main__':
    # dataset_loader = VideoDatasetLoader()
    # dataset_name = VQADataset.NextQA
    # top_k_video = 10
    # num_min_query = None
    # prep_config_set_type = PrepConfigSetType.Combination
    # # keyword_extractor_name = ''
    # keyword_extractor_name = 'Bare'
    #
    # valid_modality2config = PrepConfig.get_valid_modality_config(PrepConfig.get_valid_modality_combinations(),
    #                                                              dataset_loader.get_dataset_prep_config_set(
    #                                                                  dataset_name, type=prep_config_set_type))
    # video_id2questions, dataset_path, question_type, video_id2name = dataset_loader.load_dataset(dataset_name,
    #                                                                                              top_k_video=top_k_video,
    #                                                                                              num_min_query=num_min_query)

    dataset_names = [
        VQADataset.MSVD_QA,
        # VQADataset.NextQA,
        # VQADataset.SampleVideo,
    ]
    llm_name = LLMName.LlavaVideoQwen7b
    llm_extractor_name = llm_name
    keyword_extractor_names = [
        llm_extractor_name,
        KeyWordExtractorNames.KeyBertExtractor,
        KeyWordExtractorNames.BareExtractor,
        # KeyWordExtractorNames.NonExtractor,
    ]
    pre_config_set_types = [
        PrepConfigSetType.Combination,
        # PrepConfigSetType.VisiontokenOnly,
        # PrepConfigSetType.CaptionOnly
    ]


    iterate_run(llm_name, dataset_names, pre_config_set_types, keyword_extractors, evaluate_dataset)
