import socket
import pickle
import os

from util.preprocess_config import PrepConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time

import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import ffmpeg

modality = 'asr'
overwrite_config = {}


def cold_start():
    pass


def check_config_consistency(frame_config, ocr_config):
    assert frame_config['num_frame'] == ocr_config['num_frame']
    assert frame_config['sampling_method'] == ocr_config['sampling_method']


model_name = 'whisper-small'
whisper_model = None
whisper_processor = None
def load_whisper_model():
    global model_name, whisper_model, whisper_processor
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/{model_name}",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    whisper_processor = WhisperProcessor.from_pretrained(f"openai/{model_name}")

load_whisper_model()

def extract_audio(video_path, audio_path, acodec='pcm_s16le', ac=1, ar='16k'):
    # if not os.path.exists(audio_path):
    ffmpeg.input(video_path).output(audio_path, acodec=acodec, ac=ac, ar=ar).overwrite_output().run()


def chunk_audio(audio_path, chunk_length=30):
    speech, sr = torchaudio.load(audio_path)
    speech = speech.mean(dim=0)
    speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)
    num_samples_per_chunk = chunk_length * 16000
    chunks = []
    for i in range(0, len(speech), num_samples_per_chunk):
        chunks.append(speech[i:min(i + num_samples_per_chunk, len(speech))])
    return chunks


def transcribe_chunk(chunk):
    inputs = whisper_processor(chunk, return_tensors="pt")
    inputs["input_features"] = inputs["input_features"].to(whisper_model.device, torch.float16)
    with torch.no_grad():
        predicted_ids = whisper_model.generate(
            inputs["input_features"],
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


def get_asr_docs(video_path, audio_path, config):
    full_transcription = []
    try:
        # extract_audio(video_path, audio_path, acodec = config['acodec'], ac=config['ac'], ar=config['ar'])
        extract_audio(video_path, audio_path)
    except:
        return full_transcription

    # audio_chunks = chunk_audio(audio_path, chunk_length_s=30)
    audio_chunks = chunk_audio(audio_path, chunk_length=config['chunk_length'])

    for chunk in audio_chunks:
        transcription = transcribe_chunk(chunk)
        full_transcription.append(transcription)

    return full_transcription


def asr_whisper(video_path, audio_path, config):
    global model_name
    if config.model != model_name:
        model_name = config.model
        load_whisper_model()

    start_time = time.time()
    asr = get_asr_docs(video_path, audio_path, config)
    end_time = time.time()
    processing_time = end_time - start_time

    return {
        'asr': asr,
        "processing_time": processing_time,
        "config": config,
    }


force_overwrite = False

def warm_up():
    print('Start warm up with a sample video')
    asr_config = PrepConfig.asr_default_config()
    print(asr_whisper(warm_up_video_path, audio_path='../audio_trashbin/trash.wav', config = asr_config))
    print('Finish warm up with a sample video')

def main():
    # Set up a socket server
    port = server_config[modality]['port']
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen()
    warm_up()
    print(modality + " server is listening...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        data = client_socket.recv(8192)

        # Deserialize datas

        dataset_name, video_name, config = pickle.loads(data)
        if dataset_name is None and video_name is None and config is None:
            print(f'{modality} Server closed!')
            break

        store_path = preprocess_store_path(config['modality'], config, video_name, dataset_name)

        if not os.path.exists(store_path) or force_overwrite:
            video_path = get_video_path(dataset_name, video_name)
            assert config['modality'] == modality

            result = asr_whisper(video_path, audio_path=store_path + ".wav", config=config)

            pickle.dump(result, open(store_path, 'wb'))
            del result
        # Send back result
        client_socket.send(pickle.dumps(store_path))
        client_socket.close()


if __name__ == "__main__":
    main()
