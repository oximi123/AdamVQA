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
2. Please download the following dataset and put them into the `data/` directory.
- MSVD-QA
- MSVRTT-QA
- EgoTimeVQA
3. Run the following script to launch the modality extracting server and extract modality from the videos.
```
python preprocess_servers/frame_server.py
python preprocess_servers/caption_server.py
python preprocess_servers/visiontoken_server.py
python preprocess_servers/ocr_server.py
python preprocess_servers/object_server.py
python preprocess_servers/clip_server.py
python script_preprocess.py
```

4. Run the following script to start the tuning.
```
python main.py
```
## Citation format (TBD)

