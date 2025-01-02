#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/11/4 23:04
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import json
import logging
import uuid

import coloredlogs
import uvicorn

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from torchvision import transforms

sys.path.append(os.path.dirname(sys.path[0]))
from config.version import get_version_banner
from tools.generate import Generator, init_generate_args
from utils import save_images
from utils.processing import image_to_base64

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")
app = FastAPI()


@app.get("/")
def index():
    logger.info(msg="Route -> Hello IDDM")
    return "Hello, IDDM!"


@app.post("/api/generate/df")
def generate_diffusion_model_api(data: dict):
    """
    Generate a diffusion model
    """
    logger.info(msg="Route -> /api/df")
    logger.info(msg=f"Send json: {data}")

    # Sample type
    sample = data["sample"]
    # Image size
    image_size = data["image_size"]
    # Number of images
    num_images = data["num_images"] if data["num_images"] >= 1 else 1
    # Weight path
    weight_path = data["weight_path"]
    result_path = data["result_path"]
    # Recommend use base64 in server app
    # Return mode, base64 or url
    re_type = data["type"]

    logger.info(msg="[Web]: Start generation.")
    # Type is url or base64
    re_json = {"image": [], "type": str(re_type)}

    if any(param is None for param in [sample, image_size, num_images, weight_path, result_path, re_type]):
        return JSONResponse({"code": 400, "msg": "Illegal parameters.", "data": None})

    # Init args
    args = init_generate_args()
    args.sample = sample
    args.image_size = image_size
    args.weight_path = weight_path
    args.result_path = result_path
    # Only generate 1 image per
    args.num_images = 1

    try:
        # Init server model
        server_model = Generator(gen_args=args, deploy=True)

        logger.info(msg=f"[Web]: A total of {num_images} images are generated.")
        # Generate images by diffusion models
        for i in range(num_images):
            logger.info(msg=f"[Web]: Current generate {i + 1} of {num_images}.")
            # Generation name
            generate_name = uuid.uuid1()
            # Generate image
            x = server_model.generate(index=i)

            # Select mode
            # Recommend use base64
            if re_type == "base64":
                x = transforms.ToPILImage()(x[0])
                re_x = image_to_base64(image=x)
            else:
                # Save images
                re_x = os.path.join(result_path, f"{generate_name}.png")
                save_images(images=x, path=re_x)
            # Append return json
            image_json = {"image_id": str(generate_name), "type": re_type,
                          "image": str(re_x)}
            re_json["image"].append(image_json)

        logger.info(msg="[Web]: Finish generation.")

        return JSONResponse({"code": 200, "msg": "success!", "data": json.dumps(re_json, ensure_ascii=False)})
    except Exception as e:
        return JSONResponse({"code": 500, "msg": str(e), "data": None})


@app.post("/api/generate/sr")
def generate_super_resolution_model_api():
    logger.info(msg="Route -> /api/sr")
    # TODO: super resolution api
    return "SR!"


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 12341
    logger.info(msg=f"Run -> {host}:{port}")
    get_version_banner()
    uvicorn.run(app=app, host=host, port=port)
