#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Copyright 2025 IDDM Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    @Date   : 2024/11/4 23:04
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import json
import uuid

import uvicorn

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from torchvision import transforms

sys.path.append(os.path.dirname(sys.path[0]))
from iddm.config.version import get_version_banner
from iddm.tools.generate import Generator, init_generate_args
from iddm.utils import save_images
from iddm.utils.processing import image_to_base64
from iddm.utils.logger import get_logger

logger = get_logger(name=__name__)
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

    # Latent
    latent = data.get("latent", False)
    # Sample type
    sample = data.get("sample", "ddpm")
    # Image size
    image_size = 512 if latent else data.get("image_size", 64)
    # Number of images
    num_images = data.get("num_images") if data.get("num_images", 1) > 1 else 1
    # Use ema
    use_ema = data.get("use_ema", False)
    # Weight path
    weight_path = data.get("weight_path", None)
    result_path = data.get("result_path", "./results")
    # Autoencoder weight path
    autoencoder_ckpt = data.get("autoencoder_ckpt", None)
    # Recommend use base64 in server app
    # Return mode, base64 or url
    re_type = data.get("type", None)

    logger.info(msg="[Web]: Start generation.")
    # Type is url or base64
    re_json = {"image": [], "type": str(re_type)}

    if any(param is None for param in [sample, image_size, num_images, weight_path, result_path, re_type]):
        return JSONResponse({"code": 400, "msg": "Illegal parameters.", "data": None})

    # Init args
    args = init_generate_args()
    args.sample = sample
    args.image_size = image_size
    args.use_ema = use_ema
    args.weight_path = weight_path
    args.result_path = result_path
    args.latent = latent
    args.autoencoder_ckpt = autoencoder_ckpt
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
        logger.exception("Exception occurred while generating diffusion model")
        return JSONResponse({"code": 500, "msg": "An internal error has occurred.", "data": None})


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
