import socket
import pickle
from ape_api import setup_cfg, ape_inference, VisualizationDemo

def main():
    # Initialize demo once
    cfg = setup_cfg()
    demo = VisualizationDemo(cfg, args=None)

    # Set up a socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 9999)) 
    server_socket.listen()

    print("Server is listening...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        data = client_socket.recv(8192)
        if not data:
            break

        # Deserialize datas
        input_files, text_prompt = pickle.loads(data)
        
        # Run inference
        result = ape_inference(input_files, text_prompt, demo)
        
        # Send back result
        client_socket.send(pickle.dumps(result))
        client_socket.close()

if __name__ == "__main__":
    main()
