from util.util import *
import socket
import pickle
import time
import os
import sys
from pathlib import Path
from util.util import get_video_path

sys.path.append(str(Path('preprocess_servers/Katna').parent))
from preprocess_servers.Katna.video import Video

modality = 'frame'
overwrite_config = {}

def cold_start():
    pass
def uniform_sampling(video_path, config):
    print('---------uniform sampling----------')
    start_time = time.monotonic()

    num_frame = config['num_frame']
    assert num_frame > 0 and num_frame < 65
    vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
    total_frame_num = len(vr)
    duration = total_frame_num / vr.get_avg_fps()
    interval = round(total_frame_num / num_frame)
    frame_idx = [i for i in range(0, len(vr), interval)]
    frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    # frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    end_time = time.monotonic()
    processing_time = end_time - start_time

    return {
        "frames": spare_frames[0:num_frame],
        'frame_idx': frame_idx,
        "frame_times": frame_time[0:num_frame],
        # "video_meta": {
        #     "duration": duration,
        #     "fps": vr.get_avg_fps()
        # },
        "processing_time": processing_time,
        "config": config,
    }


def key_frame_sampling(video_path, config):
    print('---------keyframe sampling----------')
    start_time = time.monotonic()

    num_frame = config['num_frame']
    assert num_frame > 0 and num_frame < 65
    vd = Video()


    vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
    total_frame_num = len(vr)
    duration = total_frame_num / vr.get_avg_fps()
    if duration < 1200:
        print('Short video key frame sampling.')
        key_frame, frame_idx = vd.extract_video_keyframes(
            no_of_frames=num_frame, file_path=video_path
        )

    else:
        print('Long video key frame sampling.')
        key_frame, frame_idx = vd.extract_video_keyframes_big_video(
            no_of_frames=num_frame, file_path=video_path
        )
    frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    end_time = time.monotonic()
    processing_time = end_time - start_time
    return {
        "frames": spare_frames[0:num_frame],
        'frame_idx': frame_idx,
        "frame_times": frame_time[0:num_frame],
        # "video_meta": {
        #     "duration": duration,
        #     "fps": vr.get_avg_fps()
        # },
        "processing_time": processing_time,
        "config": config,
    }

force_overwrite = False

sampling_methods = {
    'uniform': uniform_sampling,
    'keyframe': key_frame_sampling,
}

from omegaconf import OmegaConf
def main():
    # Set up a socket server

    port = server_config[modality]['port']
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen()

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
            assert config['modality'] == 'frame'
            result = sampling_methods[config['sampling_method']](video_path, config)
            pickle.dump(result, open(store_path, 'wb'))
        # Send back result
        client_socket.send(pickle.dumps(store_path))
        client_socket.close()


if __name__ == "__main__":
    main()
