import socket
import pickle
import os
import time

import easyocr

from util.preprocess_config import PrepConfig


def cold_start():
    pass
modality = 'ocr'
overwrite_config = {}

def check_config_consistency(frame_config, ocr_config):
    assert frame_config['num_frame'] == ocr_config['num_frame']
    assert frame_config['sampling_method'] == ocr_config['sampling_method']


def ocr_easyocr(frames, config):
    start_time = time.time()
    reader = easyocr.Reader(['en'])
    text_set = []
    ocr_docs = []
    for img in frames:
        ocr_results = reader.readtext(img)
        det_info = ""
        for result in ocr_results:
            text = result[1]
            confidence = result[2]
            if confidence > 0.5 and text not in text_set:
                det_info += f"{text}; "
                text_set.append(text)
        if len(det_info) > 0:
            ocr_docs.append(det_info)
    end_time = time.time()
    processing_time = end_time - start_time

    return {
        'ocr': ocr_docs,
        "processing_time": processing_time,
        "config": config,
    }

def warm_up():
    print('Start warm up with a sample video')
    ocr_easyocr([np.random.rand(200, 200, 3).astype(np.float32)  for _ in range(10)], PrepConfig.ocr_default_config())
    print('Finish warm up with a sample video')

force_overwrite = False
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
        video_path = get_video_path(dataset_name, video_name)
        assert config['modality'] ==  modality

        frame_path = preprocess_store_path('frame', config, video_name, dataset_name)
        store_path = preprocess_store_path(config['modality'], config, video_name, dataset_name)

        if not os.path.exists(store_path) or force_overwrite:
            frame_result = pickle.load(open(frame_path, 'rb'))
            check_config_consistency(frame_result['config'], config)

            frames = frame_result['frames']
            result = ocr_easyocr(frames, config)

            pickle.dump(result, open(store_path, 'wb'))
        # Send back result
        client_socket.send(pickle.dumps(store_path))
        client_socket.close()


if __name__ == "__main__":
    main()
