# AdamVQA: Efficient Modality Adaptation for SLO-aware LLM-based Video Question Answering

This repository contains the core code of paper: AdamVQA: Efficient Modality Adaptation for SLO-aware LLM-based Video Question Answering

## Introduction

AdamVQA is a Efficient Modality Adaptation for SLO-aware LLM-based Video Question Answering

- An offline tuning module leverages inter-video similarity to tuning modality combinations and knobs for videos and questions.
- A online adaptation module that leverages inter-question similarity to ensemble best configuration for VQA.
## Usage
1. Please run the following command to install necessary
```
pip install -r requirements.txt
```
2. Download the following dataset and put them into the `data/` directory, and put the video files in to `{dataset}/videos`.
- MSVD-QA
- MSVRTT-QA
- EgoTimeVQA

3. Download the models from the following repositories and put them into `model/` directory:
```
https://github.com/YueFan1014/VideoAgent
https://github.com/Leon1207/Video-RAG-master
https://huggingface.co/lmsys/vicuna-13b-v1.5
```
4. Run the following script to launch the modality extracting server and extract modality from the videos.
```
python preprocess_servers/frame_server.py
python preprocess_servers/caption_server.py
python preprocess_servers/visiontoken_server.py
python preprocess_servers/ocr_server.py
python preprocess_servers/object_server.py
python preprocess_servers/clip_server.py
python script_preprocess.py
```
5. Serve the `vicuna-13b-v1.5` model using the `vLLM` framework and install `FastChat`. You need to create another environment for `vLLM`.
```
conda activate vllm
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
python3 -m fastchat.serve.controller
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.vllm_worker --model-path /home/ubuntu/model/vicuna-13b-v1.5 --gpu_memory_utilization 0.8
```
6. Run the following script to start the tuning.
```
python main.py
```
## Citation format (TBD)

