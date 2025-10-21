import socket
import pickle
import os
import time

from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM

from util.preprocess_config import PrepConfig
from videollama2 import model_init
from videollama2.utils import disable_torch_init

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda"
overwrite_config = {}
modality = 'visiontoken'

def cold_start():
    pass

def check_config_consistency(frame_config, visiontoken_config):
    assert frame_config['num_frame'] == visiontoken_config['num_frame']
    assert frame_config['sampling_method'] == visiontoken_config['sampling_method']

def llava_visiontoken(frames, config):
    start_time = time.monotonic()

    video = processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda(0).bfloat16()
    video = [video]
    visiontoken = model.image2token(video)

    end_time = time.monotonic()
    processing_time = end_time - start_time

    return {
        'visiontoken': visiontoken,
        "processing_time": processing_time,
        "config": config,
    }

def llama_visiontoken(frames, config):
    disable_torch_init()

    modal = 'video'
    time_start = time.monotonic()
    video = processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
    video = [(video, modal)]
    visiontoken = model.image2token(video)
    time_end = time.monotonic()
    processing_time = time_end - time_start
    return {
        'visiontoken': visiontoken,
        "processing_time": processing_time,
        "config": config,
    }



def warm_up_llava():
    print('Start warm up with a sample video')
    llava_visiontoken([np.random.random((200, 200, 3)) for _ in range(10)], PrepConfig.visiontoken_default_config())
    print('Finish warm up with a sample video')

def warm_up_llama():
    print('Start warm up with a sample video')
    llama_visiontoken([np.random.random((200, 200, 3)) for _ in range(10)], PrepConfig.visiontoken_default_config())
    print('Finish warm up with a sample video')

def load_model(llm_name):
    global tokenizer, model, processor
    if model is not None:
        return
    if llm_name == LLMName.LlavaVideoQwen7b:
        tokenizer, model, processor, max_length = load_pretrained_model(
            "/home/ubuntu/model/LLaVA-Video-7B-Qwen2",
            None,
            "llava_qwen",
            torch_dtype="bfloat16",
            device_map="auto",
            overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args
        assert isinstance(model, LlavaQwenForCausalLM)
        model.eval()

        warm_up_llava()
    elif llm_name == LLMName.VideoLlamaMistral7b:
        model_path = '/home/ubuntu/model/VideoLLaMA2-7B-16F'
        model, processor, tokenizer = model_init(model_path, num_frame=4)
        warm_up_llama()
    else:
        raise Exception("Unknown llm model")
force_overwrite = False
tokenizer, model, processor = None, None, None
def main():
    # Set up a socket server

    port = server_config[modality]['port']
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen()
    # Initialize demo once


    print(modality + " server is listening...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        data = client_socket.recv(8192)
        if not data:
            break

        # Deserialize datas
        dataset_name, video_name, config, llm_name = pickle.loads(data)
        load_model(llm_name)
        if dataset_name is None and video_name is None and config is None:
            print(f'{modality} Server closed!')
            break

        store_path = preprocess_store_path(config['modality'], config, video_name, dataset_name, llm_name)

        if not os.path.exists(store_path) or force_overwrite:
            assert config['modality'] == modality
            frame_path = preprocess_store_path('frame', config, video_name, dataset_name, llm_name)
            frame_result = pickle.load(open(frame_path, 'rb'))
            check_config_consistency(frame_result['config'], config)


            frames = frame_result['frames']

            # Run inference
            if llm_name == LLMName.LlavaVideoQwen7b:
                result = llava_visiontoken(frames, config)
            elif llm_name == LLMName.VideoLlamaMistral7b:
                result = llama_visiontoken(frames, config)

            pickle.dump(result, open(store_path, 'wb'))
            del result
        # Send back result
        client_socket.send(pickle.dumps(store_path))
        client_socket.close()


if __name__ == "__main__":
    main()
