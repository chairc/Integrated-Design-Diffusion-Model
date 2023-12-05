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

import torch

sys.path.append(os.path.dirname(sys.path[0]))
from model.networks.unet import UNet
from utils.utils import save_images
from utils.initializer import device_initializer, sample_initializer
from utils.checkpoint import load_ckpt

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def generate(parse_json_data):
    """
    Generating (deploy version)
    :param parse_json_data: Parse send json message
    :return: JSON
    """
    logger.info(msg="[Client]: Start generation.")
    re_json = {"image": []}
    # Get the incoming json value information
    # Enable conditional generation
    conditional = parse_json_data["conditional"]
    # Sample type
    sample = parse_json_data["sample"]
    # Image size
    image_size = parse_json_data["image_size"]
    # Number of images
    num_images = parse_json_data["num_images"] if parse_json_data["num_images"] >= 1 else 1
    # Activation function
    act = parse_json_data["act"]
    # Weight path
    weight_path = parse_json_data["weight_path"]
    # Saving path
    result_path = parse_json_data["result_path"]
    # Run device initializer
    device = device_initializer()
    # Initialize the diffusion model
    diffusion = sample_initializer(sample=sample, image_size=image_size, device=device)
    # Initialize model
    if conditional:
        # Number of classes
        num_classes = parse_json_data["num_classes"]
        # Generation class name
        class_name = parse_json_data["class_name"]
        # classifier-free guidance interpolation weight
        cfg_scale = parse_json_data["cfg_scale"]
        model = UNet(num_classes=num_classes, device=device, image_size=image_size, act=act).to(device)
        load_ckpt(ckpt_path=weight_path, model=model, device=device, is_train=False)
        y = torch.Tensor([class_name]).long().to(device)
    else:
        model = UNet(device=device, image_size=image_size, act=act).to(device)
        load_ckpt(ckpt_path=weight_path, model=model, device=device, is_train=False)
        y = None
        cfg_scale = None
    # Generate images by diffusion models
    for i in range(num_images):
        # Generation name
        generate_name = uuid.uuid1()
        x = diffusion.sample(model=model, n=1, labels=y, cfg_scale=cfg_scale)
        # TODO: Convert to base64
        # Save images
        save_images(images=x, path=os.path.join(result_path, f"{generate_name}.jpg"))
        # Append return json
        image_json = {"image_id": str(generate_name),
                      "image_name": f"{generate_name}.jpg"}
        re_json["image"].append(image_json)
    logger.info(msg="[Client]: Finish generation.")
    return re_json


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
    main()
