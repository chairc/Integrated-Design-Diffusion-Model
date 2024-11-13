#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/8/9 1:09
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import json
import uuid
import socket
import logging
import threading
import coloredlogs

from torchvision import transforms

sys.path.append(os.path.dirname(sys.path[0]))
from tools.generate import init_generate_args, Generator
from utils.utils import save_images
from utils.processing import image_to_base64
from config.version import get_version_banner

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def generate(parse_json_data):
    """
    Generating (deploy version)
    :param parse_json_data: Parse send json message
    :return: JSON
    """
    # Sample type
    sample = parse_json_data["sample"]
    # Image size
    image_size = parse_json_data["image_size"]
    # Number of images
    num_images = parse_json_data["num_images"] if parse_json_data["num_images"] >= 1 else 1
    # Weight path
    weight_path = parse_json_data["weight_path"]
    result_path = parse_json_data["result_path"]
    # Return mode, base64 or url
    re_type = parse_json_data["type"]

    logger.info(msg="[Client]: Start generation.")
    # Type is url or base64
    re_json = {"image": [], "type": str(re_type)}

    # Init args
    args = init_generate_args()
    args.sample = sample
    args.image_size = image_size
    args.weight_path = weight_path
    args.result_path = result_path
    # Only generate 1 image per
    args.num_images = 1

    # Init model
    model = Generator(gen_args=args, deploy=True)

    logger.info(msg=f"[Client]: A total of {num_images} images are generated.")
    # Generate images by diffusion models
    for i in range(num_images):
        logger.info(msg=f"[Client]: Current generate {i + 1} of {num_images}.")
        # Generation name
        generate_name = uuid.uuid1()
        x = model.generate(index=i)
        # Select mode
        if re_type == "base64":
            x = transforms.ToPILImage()(x[0])
            re_x = image_to_base64(image=x)
        else:
            # Save images
            re_x = os.path.join(result_path, f"{generate_name}.png")
            save_images(images=x, path=re_x)
        # Append return json in data
        image_json = {"image_id": str(generate_name), "type": re_type,
                      "image": str(re_x)}

        re_json["image"].append(image_json)
    logger.info(msg="[Client]: Finish generation.")
    return json.dumps(re_json, ensure_ascii=False)


def main():
    """
    Main function
    """
    # Create server socket
    server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
    # Get localhost name
    host = socket.gethostname()
    # Set port
    port = 12345
    # Bind the socket with localhost and port
    server_socket.bind((host, port))
    # Set the maximum number of listener
    server_socket.listen(5)
    # Get the connection information of the local server
    local_server_address = server_socket.getsockname()
    logger.info(msg=f"[Server]: Server address: {str(local_server_address)}")
    # Loop waiting to receive client message
    while True:
        # Get a client connection
        client_socket, address = server_socket.accept()
        logger.info(msg=f"[Server]: Connection address: {str(address)}")
        try:
            # Start a processing thread for each request
            t = ServerThreading(client_socket=client_socket, address=address)
            t.start()
        except Exception as identifier:
            logger.error(msg=f"[Server]: [Error] {identifier}")
            break
    server_socket.close()
    logger.info(msg=f"[Server]: Server close: {str(local_server_address)}")


class ServerThreading(threading.Thread):
    """
    ServerThreading class
    """

    def __init__(self, client_socket, address, receive_size=1024 * 1024, encoding="utf-8"):
        """
        ServerThreading initialization
        :param client_socket: Client socket
        :param address: Client address
        :param receive_size: Receive size
        :param encoding: Encoding type
        """
        threading.Thread.__init__(self)
        self.socket = client_socket
        self.address = address
        self.receive_size = receive_size
        self.encoding = encoding

    def run(self):
        """
        Run ServerThreading
        """
        logger.info(msg=f"[Client]: {self.address} start a new thread.")
        try:
            # Receive message
            msg = ""
            while True:
                # Read the byte of receive_size
                rec = self.socket.recv(self.receive_size)
                # Decode
                msg += rec.decode(self.encoding)
                # Whether the text acceptance is complete,
                # because the python socket cannot judge whether the received data is complete,
                # so it is necessary to customize the protocol to mark the data acceptance complete
                if msg.strip().endswith("-iccv-over"):
                    msg = msg[:-10]
                    break
            # Parse data in json format
            parse_json_data = json.loads(s=msg)
            # Generate images
            send_msg = generate(parse_json_data=parse_json_data)
            logger.info(msg=f"[Client]: [Successfully] Current address id: {self.address}, return message: {send_msg}")
            # Send message
            self.socket.send(f"{send_msg}".encode(self.encoding))
        except Exception as identifier:
            self.socket.send("500".encode(self.encoding))
            logger.error(msg=f"[Client]: [Error] {identifier}")
        finally:
            self.socket.close()
        logger.info(msg=f"[Client]: {self.address} finish the thread.")

    def __del__(self):
        """
        Destroy object
        :return: None
        """
        pass


if __name__ == "__main__":
    get_version_banner()
    main()
