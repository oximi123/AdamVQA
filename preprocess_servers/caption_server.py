import sys

from util.preprocess_config import PrepConfig
from util.preprocess_constant import CaptionModelName

sys.path.insert(0, 'LaViLa/')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import urllib.request
from collections import OrderedDict
import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

from LaViLa.lavila.data.video_transforms import Permute
from LaViLa.lavila.models.models import VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL
from LaViLa.lavila.models.tokenizer import MyGPT2Tokenizer
from LaViLa.eval_narrator import decode_one
import cv2

import socket
import pickle
import os
import time

import easyocr
import PIL.Image as Image

modality = 'caption'
overwrite_config = {}
def caption_llava(video_path, config, pipe):
    assert config['frames_per_caption'] == 1, 'llava model only support single image caption!'
    start_time = time.monotonic()
    seconds_per_caption = config['seconds_per_caption']
    vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
    frame_interval = int(seconds_per_caption * vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), frame_interval)]
    frames = vr.get_batch(frame_idx).asnumpy()
    # vidcap = cv2.VideoCapture(video_path)
    # count = 0
    # success = True
    # fps_ori = int(vidcap.get(cv2.CAP_PROP_FPS))
    # frame_interval = seconds_per_caption * fps_ori
    # while success:
    #     success, image = vidcap.read()
    #     if not success:
    #         break
    #     if count % (frame_interval) == 0 :
    #         # cv2.imwrite(f'{output_path}/{count}.jpg', image)
    #         images.append(image)
    #     count+=1

    # hparams and model init
    max_token = 100
    num_words = 10
    model_id = "/home/ubuntu/model/llava-1.5-7b-hf"
    # pipe = pipeline("image-to-text", model=model_id, device=1)
    prompt = f"Describe the image in {num_words} words."
    prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

    # get subset uids

    # output_base_path = Path(f'/mnt/opr/ce/datasets/EgoSchema/captions/llava/subset/words_{num_words}')
    # output_base_path.mkdir(parents=True, exist_ok=True)

    # example_output_path = output_base_path / f'{images_dir.name}.json'
    captions = dict()
    for id, image in enumerate(frames):
        image = Image.fromarray(image)
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_token})
        outputs = str(outputs[0]['generated_text'])
        idx = outputs.find("ASSISTANT")
        text = outputs[idx+11:]
        print(text)
        caption_start_frame = id * frame_interval
        caption_end_frame = (id + 1) * frame_interval
        segment = "{}_{}".format(str(caption_start_frame), str(caption_end_frame))
        captions[segment] = text

    segments = list(captions)
    segment2id = dict()
    for segment in segments:
        segment2id[segment] = len(segment2id)

    end_time = time.monotonic()
    processing_time = end_time - start_time

    return {
        'caption': captions,
        'segment2id': segment2id,
        "processing_time": processing_time,
        "config": config,
    }

def caption_lavila(video_path, config, model, tokenizer):
    """create the captions for all videos"""
    start_time = time.monotonic()
    crop_size = 336
    val_transform = transforms.Compose([
        Permute([3, 0, 1, 2]),
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001],
                                        std=[68.5005327, 66.6321579, 70.32316305])
    ])

    seconds_per_caption = config['seconds_per_caption']
    frames_per_caption = config['frames_per_caption']
    assert config['modality'] == modality

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_captions = total_frames // (fps * seconds_per_caption)
    frame_interval = fps * seconds_per_caption // frames_per_caption  # the interval between two selected frames

    captions = dict()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for caption_id in range(total_captions):
        frames = []
        for i in range(frames_per_caption):  # 4 frames are selected for generating the caption
            success, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            for j in range(frame_interval - 1):  # skip other frames
                success, frame = cap.read()
        for i in range(fps * seconds_per_caption - frame_interval * frames_per_caption):
            success, frame = cap.read()  # skip remaining frames
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
        frames = torch.stack(frames, dim=0)
        frames = val_transform(frames)
        frames = frames.unsqueeze(0)
        num_return_sequences = 5 if 'num_return_sequences' not in config.keys() or config[
            'num_return_sequences'] is None else config['num_return_sequences']
        temperature = 0.7 if 'temperature' not in config.keys() or config['temperature'] is None else config[
            'temperature']
        with torch.no_grad():
            input_frames = frames.cuda(non_blocking=True)
            image_features = model.encode_image(input_frames)
            generated_text_ids, ppls = model.generate(
                image_features,
                tokenizer,
                target=None,  # free-form generation
                max_text_length=77,
                top_k=None,
                top_p=0.95,  # nucleus sampling
                num_return_sequences=num_return_sequences,  # number of candidates: num_return_sequences
                temperature=temperature,
                early_stopping=True,
            )
        del input_frames
        del image_features
        del frames

        text = ""
        ppl = -1
        assert len(ppls) == num_return_sequences
        for i in range(num_return_sequences):  # ppl最大的sequence作为结果
            generated_text_str = decode_one(generated_text_ids[i], tokenizer)
            if ppls[i] > ppl:
                ppl = ppls[i]
                text = generated_text_str
        caption_start_frame = caption_id * fps * seconds_per_caption
        caption_end_frame = (caption_id + 1) * fps * seconds_per_caption
        segment = "{}_{}".format(str(caption_start_frame), str(caption_end_frame))
        captions[segment] = text
        print(f"id: {caption_id}, frame_interval: {segment}, caption: {text}")
    cap.release()
    segments = list(captions)
    segment2id = dict()
    for segment in segments:
        segment2id[segment] = len(segment2id)

    end_time = time.monotonic()
    processing_time = end_time - start_time

    return {
        'caption': captions,
        'segment2id': segment2id,
        "processing_time": processing_time,
        "config": config,
    }

def warm_up(model, tokenizer):
    print('Start warm up lavila with a sample video')
    caption_lavila(warm_up_video_path, PrepConfig.caption_default_config(), model, tokenizer)
    print('Finish warm up lavila with a sample video')
force_overwrite = False

from transformers import pipeline
def  load_llava_pipe():
    pipe = pipeline("image-to-text", 'llava-hf/llava-1.5-7b-hf',revision='a272c74', device='cuda')
    return pipe

def load_lavila_model():
    ckpt_name = 'vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth'
    ckpt_path = os.path.join(os.getenv('HOME'), 'model', ckpt_name)
    if not os.path.exists(ckpt_path):
        print('downloading model to {}'.format(ckpt_path))
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/{}'.format(ckpt_name),
                                   ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    # instantiate the model, and load the pre-trained weights
    model = VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL(
        text_use_cls_token=False,
        project_embed_dim=256,
        gated_xattn=True,
        timesformer_gated_xattn=False,
        freeze_lm_vclm=False,  # we use model.eval() anyway
        freeze_visual_vclm=False,  # we use model.eval() anyway
        num_frames=4,
        drop_path_rate=0.
    )
    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()
    tokenizer = MyGPT2Tokenizer('gpt2-xl', add_bos=True)
    return model, tokenizer

model, tokenizer = load_lavila_model()
pipe = load_llava_pipe()

def main():


    # Set up a socket server
    port = server_config[modality]['port']
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen()

    # warm_up(model, tokenizer)
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

            caption_model = config['model']
            if caption_model == CaptionModelName.lavila:
                result = caption_lavila(video_path, config, model, tokenizer)
            elif caption_model == CaptionModelName.llava:
                result = caption_llava(video_path, config, pipe)
            else:
                raise NotImplementedError(f'Model {caption_model} not supported.')

            pickle.dump(result, open(store_path, 'wb'))
            del result
        # Send back result
        client_socket.send(pickle.dumps(store_path))
        client_socket.close()


if __name__ == "__main__":
    main()
