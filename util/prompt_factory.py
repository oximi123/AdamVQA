from string import Template

from preprocess_constant import QuestionType, Modality, VQADataset
from util import caption2prompt


class PromptTemplate(object):
    def __init__(self, template, post_process_fn, head=None):
        self.head = head
        self.prompt_template = template
        self.post_process_fn = post_process_fn

    def get_num_stages(self):
        return len(self.template)

    def get_template_str(self):
        template = []
        for temp in self.prompt_template:
            template.append(temp.safe_substitute())
        return template

    def fill(self, **kwargs):
        # match variable names: duration, narration, question, optionA, optionB, optionC, optionD, optionE, num_words
        prompt_filled = []
        for temp in self.prompt_template:
            prompt_filled.append(temp.substitute(kwargs))
        return prompt_filled if len(prompt_filled) > 1 else prompt_filled[0]


import itertools


class PromptFactory:

    def __init__(self):
        self.modstr2prompt = {}

        self.question_types = [QuestionType.OE, QuestionType.MC]

        # todo 加入multi choice 问题的prompt

    def process_video_meta(self, video_meta: dict):
        video_meta['duration'] = round(video_meta['duration'], 1)
        return video_meta

    def process_MC_question(self, dataset_name, question_dict: dict):
        pass

    def caption_only_prompt_OE(self, **kwargs):
        prompt = Template(
            "You are given some language descriptions of a video. "
            "The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. "
            "The descriptions are sequential and non-overlapping which cover the whole video exactly. "
            "Here are the descriptions:\n$caption\n "
            "You are going to answer an open-ended question based on the descriptions.\n"
            "Here is the question: $question.").substitute(kwargs)
        return prompt

    def caption_only_prompt_MC(self, dataset_name, **kwargs):
        if dataset_name == VQADataset.ALLVB:
            prompt = Template(
                "Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, "
                "and your answer must be one of the letters (A, B, C, D, or E). "
                "You must not provide any other response or explanation. "
                "You are given some language descriptions of a first person view video. "
                "The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. "
                "The descriptions are sequential and non-overlapping which cover the whole video exactly. "
                "Here are the descriptions: $caption\n "
                "You are going to answer a multiple choice question based on the descriptions, "
                "and your answer should be a single letter chosen from the choices.\n Here is the question: $org_question.\n "
                "Here are the choices:\n A: $optionA\n B: $optionB\n C: $optionC\n D: $optionD\n E: $optionE\n\n "
                "In your response, the first character should be your answer to this multiple choice question.").substitute(
                kwargs)
        elif dataset_name == VQADataset.NextQA or dataset_name == VQADataset.SampleVideo:
            prompt = Template(
                "Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, "
                "and your answer must be one of the letters (A, B, C, D, or E). "
                "You must not provide any other response or explanation. "
                "You are given some language descriptions of a first person view video. "
                "The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. "
                "The descriptions are sequential and non-overlapping which cover the whole video exactly. "
                "Here are the descriptions: $caption\n "
                "You are going to answer a multiple choice question based on the descriptions, "
                "and your answer should be a single letter chosen from the choices.\n Here is the question: $org_question.\n "
                "Here are the choices:\n A: $a0\n B: $a1\n C: $a2\n D: $a3\n E: $a4\n\n "
                "In your response, the first character should be your answer to this multiple choice question.").substitute(
                kwargs)
        else:
            raise NotImplementedError
        return prompt

    def visiontoken_only_prompt_OE(self, dataset_name, question):
        # return Template("$question.\nAnswer a question using a short phrase or sentence.")
        pre_prompt = ""
        post_prompt = "\nAnswer a question using a short phrase or sentence."
        return f'{pre_prompt}\n{question}\n{post_prompt}'

    def visiontoken_only_prompt_MC(self, dataset_name, question):
        if dataset_name == VQADataset.CinePipe:
            pre_prompt = "You will be provided with subtitles from a specific scene of a movie and all the video frames from that scene. After going through the movie scene and seeing the frames, please answer the question that follows. The question will have five possible answers labeled A, B, C, D, and E, please try to provide the most probable answer in your opinion. Your output should be just one of A,B,C,D,E and nothing else.\n**Output Format:**\n\t**Answer:** <Option_key>"
            post_prompt = "Note: Follow the output format strictly. Only answer with the option key (A, B, C, D, E) and nothing else."
        elif dataset_name == VQADataset.ALLVB:
            pre_prompt = ""
            post_prompt = "\nAnswer with the option's letter from the given choices directly."
        elif dataset_name == VQADataset.NextQA or dataset_name == VQADataset.SampleVideo:
            pre_prompt = ""
            post_prompt = "\nAnswer with the option's letter from the given choices directly."
        else:
            raise NotImplementedError
        prompt = f'{pre_prompt}\n{question}\n{post_prompt}'
        return prompt

    def process_caption(self, caption_dict: dict):
        caption_str = ''
        for i, (_, val) in enumerate(caption_dict.items()):
            if not val.endswith('.'):
                val += '.'
            caption_str += f'{i}: {val}\n'
        return caption_str

    def get_prompt(self, modality2doc: dict, modality2config, dataset_name, question_type, question_dict, video_meta,
                   **kwargs):
        video_meta = self.process_video_meta(video_meta)
        question = question_dict['question']
        if len(modality2doc) == 0:
            return question
        elif len(modality2doc) == 1:
            selected_modality = list(modality2doc.keys())[0]
            if selected_modality == Modality.VisionToken:
                return self.visiontoken_only_prompt_OE(dataset_name=dataset_name,
                                                       question=question) if question_type == QuestionType.OE else self.visiontoken_only_prompt_MC(
                    dataset_name=dataset_name,
                    question=question)
            elif selected_modality == Modality.Caption:
                caption_doc = self.process_caption(modality2doc[Modality.Caption])
                caption_config = modality2config[Modality.Caption]

                return self.caption_only_prompt_OE(duration=video_meta['duration'],
                                                   clip_length=caption_config['seconds_per_caption'],
                                                   caption=caption_doc,
                                                   question=question) if question_type == QuestionType.OE \
                    else self.caption_only_prompt_MC(
                    dataset_name=dataset_name,
                    duration=video_meta['duration'],
                    clip_length=caption_config['seconds_per_caption'],
                    caption=caption_doc, **question_dict)
        return self.video_rag_prompt(modality2doc, question_type, question, **kwargs)

    def video_rag_prompt(self, modality2doc: dict, question_type, question, **kwargs):
        prompt = ""
        obj_prompt = ''
        asr_prompt = ''
        ocr_prompt = ''
        caption_prompt = ''
        question_prompt = ''
        selected_modalities = modality2doc.keys()
        if Modality.Object in selected_modalities:
            assert isinstance(modality2doc[Modality.Object], dict)
            det_docs = modality2doc[Modality.Object]['det_docs']
            det_top_idx = modality2doc[Modality.Object]['det_top_idx']
            max_frames_num = modality2doc[Modality.Object]['max_frames_num']
            for i, info in enumerate(det_docs):
                if len(info) > 0:
                    prompt += f"Frame {str(det_top_idx[i] + 1)}: " + info + "\n"
                    obj_prompt += f"Frame {str(det_top_idx[i] + 1)}: " + info + "\n"
            if len(prompt) > 0:
                prompt = f"\nVideo have {str(max_frames_num)} frames in total, the detected objects' information in specific frames: \n" + prompt
                obj_prompt = f"\nVideo have {str(max_frames_num)} frames in total, the detected objects' information in specific frames: \n" + obj_prompt
        if Modality.ASR in selected_modalities:
            asr_docs = modality2doc[Modality.ASR]
            if len(asr_docs) > 0:
                prompt += "\nVideo Automatic Speech Recognition information (given in chronological order of the video): " + " ".join(
                    asr_docs) + '\n'
                asr_prompt += "\nVideo Automatic Speech Recognition information (given in chronological order of the video): " + " ".join(
                    asr_docs) + '\n'
        if Modality.OCR in selected_modalities:
            ocr_docs = modality2doc[Modality.OCR]
            if len(ocr_docs) > 0:
                prompt += "\nVideo OCR information (given in chronological order of the video): " + "; ".join(
                    ocr_docs) + '\n'
                ocr_prompt += "\nVideo OCR information (given in chronological order of the video): " + "; ".join(
                    ocr_docs) + '\n'
        if Modality.Caption in selected_modalities:
            caption_docs = modality2doc[Modality.Caption]
            if len(caption_docs) > 0:
                # caption_prompt = caption2prompt(caption_docs)
                tmp = caption2prompt(caption_docs)
                prompt += "\nVideo Caption information (given in chronological order of the video):\n" + ''.join(
                    tmp)
                caption_prompt += "\nVideo Caption information (given in chronological order of the video):\n" + ''.join(
                    tmp)

        if question_type == QuestionType.MC:
            prompt += "Select the best answer to the following multiple-choice question based on the video and the information (if given). Respond with only the letter (A, B, C, D, E or F) of the correct option.\n" + question + "\nThe best answer is:"
            question_prompt += "Select the best answer to the following multiple-choice question based on the video and the information (if given). Respond with only the letter (A, B, C, D, E or F) of the correct option.\n" + question + "\nThe best answer is:"
        else:
            prompt += "Please answer the following open-ended question based on the video and the information (if given). Respond with only the answer without any other information. Question: " + question
            question_prompt += "Please answer the following open-ended question based on the video and the information (if given). Respond with only the answer without any other information. Question: " + question

        texts = [obj_prompt, asr_prompt, ocr_prompt, caption_prompt, question_prompt]
        assert prompt == obj_prompt + asr_prompt + ocr_prompt + caption_prompt + question_prompt
        tokenize_func = kwargs.pop('tokenize_func', None)
        tokenizer = kwargs.pop('tokenizer', None)
        video_embedding = kwargs.pop('video_embedding', None)
        max_context_length = kwargs.pop('max_context_length', None)
        token_margin = kwargs.pop('token_margin', 100)  # 因为该prompt之后还要套用对应模型的模板，为此预留一些

        input_ids = tokenize_func(video_embedding, prompt).unsqueeze(
            0)
        num_input_tokens = max(input_ids.shape)
        if video_embedding is not None:
            assert len(video_embedding.shape) == 2

        len_limit = max_context_length - token_margin
        video_embedding_len = video_embedding.shape[0] if video_embedding is not None else 0
        over_flow_num = num_input_tokens + video_embedding_len - len_limit
        if over_flow_num > 0:
            text_ids = [tokenizer.encode(text) for text in texts]
            text_lens = [len(text_id) for text_id in text_ids]
            if float(max(text_lens)) / len_limit > 0.5:  # 其中一个modality占了绝大多数，只减这一个
                longest_id = text_lens.index(max(text_lens))
                longest_text_ids = text_ids[longest_id][0:-over_flow_num]
                texts[longest_id] = tokenizer.decode(longest_text_ids)
            else:  # 其他modality也占了很多，按比例减少
                raise NotImplementedError

        prompt = ''
        for text in texts:
            prompt += text
        return prompt

    def retrieve_prompt(self, question):
        prompt = "Question: " + question
        # you can change this decouple prompt to fit your requirements
        prompt += "\nTo answer the question step by step, you can provide your retrieve request to assist you by the following json format:"
        prompt += '''{
            "ASR": Optional[str]. The subtitles of the video that may be relevant to the question you want to retrieve, in two sentences. If you no need for this information, please return null.
            "DET": Optional[list]. (The output must include only physical entities, not abstract concepts, less than five entities) All the physical entities and their location related to the question you want to retrieve, not abstract concepts. If you don't need this information, please return null.
            "TYPE": Optional[list]. (The output must be specified as null or a list containing only one or more of the following strings: 'location', 'number', 'relation'. No other values are valid for this field) The information you want to obtain about the detected objects. If you need the object location in the video frame, output "location"; if you need the number of specific object, output "number"; if you need the positional relationship between objects, output "relation". 
        }
        ## Example 1: 
        Question: How many blue balloons are over the long table in the middle of the room at the end of this video? A. 1. B. 2. C. 3. D. 4.
        Your retrieve can be:
        {
            "ASR": "The location and the color of balloons, the number of the blue balloons.",
            "DET": ["blue balloons", "long table"],
            "TYPE": ["relation", "number"]
        }
        ## Example 2: 
        Question: In the lower left corner of the video, what color is the woman wearing on the right side of the man in black clothes? A. Blue. B. White. C. Red. D. Yellow.
        Your retrieve can be:
        {
            "ASR": null,
            "DET": ["the man in black", "woman"],
            "TYPE": ["location", "relation"]
        }
        ## Example 3: 
        Question: In which country is the comedy featured in the video recognized worldwide? A. China. B. UK. C. Germany. D. United States.
        Your retrieve can be:
        {
            "ASR": "The country recognized worldwide for its comedy.",
            "DET": null,
            "TYPE": null
        }
        Do not copy the content of this three examples and provide your retrieve request based on the question! Note that you don't need to answer the question in this step, so you don't need any information about the video of image. You only need to provide your retrieve request (it's optional), and I will help you retrieve the information you want. Please provide the json format.'''
        return prompt


import random


def first_char_as_answer(res):
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    if len(res) == 0:
        return -1
    if res[0] in mapping:
        return mapping[res[0]]
    return -1


def first_char_as_answer_raw(res):
    candidates = ['A', 'B', 'C', 'D']
    if len(res) == 0:
        return random.choice(candidates)
    if res[0] in candidates:
        return res[0]
    return random.choice(candidates)


def identity(res):
    return res


def first_char_after_anchor(anchor):
    def f(res):
        mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        anchor_index = res.find(anchor)
        pred = -1  # if decoding failed, return -1
        if anchor_index >= 0:
            pred_letter = res[anchor_index + len(anchor)]
            if pred_letter in mapping:
                pred = mapping[pred_letter]
        return pred

    return f


def get_intervals_as_list(text):
    text = text.split('.')[0]
    text = text.strip()
    if text[-1] != ']':
        index = text.rfind(']')
        assert index > 0
        text = text[:index + 1]
    interval_list_text = text.split('and')
    intervals = []
    for interval_text in interval_list_text:
        if ',' not in interval_text:
            intervals.append([0, 0])
            continue
        start_text, end_text = interval_text.split(',')
        start_text, end_text = start_text.strip(' []'), end_text.strip(' []')
        if start_text == 'None':
            start_text = '0'
        if end_text == 'None':
            end_text = '1'
        try:
            start, end = int(start_text), int(end_text)
        except:
            start, end = 0, 1
        intervals.append([start, end])
    return intervals
