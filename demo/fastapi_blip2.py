import sys
import os

from PIL import Image
import torch
from xllm.models import load_model_and_preprocess
from xllm.models import xllm_models

import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from PIL import Image
import random
import base64
from io import BytesIO


app = FastAPI()

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
def load_model() -> xllm_models:
    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(name="xllm_image", model_type="pretrain_xllm_image", is_eval=True, device=device)
    return model, vis_processors, device


model, vis_processors, device = load_model()


@app.post("/getChatBotResponse")
def get_bot_response(msg: dict):
    '''
        msg: {
            'image': ...,
            'text': [(A1,B1), (A2,B2), ..., (An)]
        }

        return response: str
    '''
    model.to('cuda')
    print(msg['text'])
    if len(msg['text']) > 1:
        history = msg['text'][:-1]
        query = msg['text'][-1]
    else:
        history = []
        query = msg['text'][-1]
    if not history:
        prompt = "[Round {}]\n问：{}\n".format(1, query)
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    print("prompt: ", prompt)
    if msg['image'] is not None:
      print('use image for dialog')
      use_image = True
      image = Image.open(BytesIO(base64.urlsafe_b64decode(msg['image']))).convert("RGB")
    #   image.save('images/new.png')
      image = vis_processors["eval"](image).unsqueeze(0).to(device)
    else:
      print('no image')
      use_image = False
      image = None
    output = model.generate_demo({"image": image, "prompt": prompt.strip('\n'), "use_image": use_image})[0]
   
    print(output)
    return str(output)

if __name__ == "__main__":
    # uvicorn.run("main:app")
    uvicorn.run(app="fastapi_xllm:app", host="0.0.0.0", port=12312, reload=False)
