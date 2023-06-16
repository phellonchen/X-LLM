import json
import os
from tqdm import tqdm
from PIL import Image

caption_txt = '/raid/cfl/data/wukong/caption-image-total.txt'

cap_data = open(caption_txt)


image_suffix= ['jpg', 'jpeg','png', 'JPEG', 'PNG']

data_list = []
cnt = 0
image_paths = []
for line in tqdm(cap_data.readlines()):
    try:
        line = line.strip()
        lines = line.split('\t')
        image_path = str(lines[1])
        


        image_path1 = os.path.join('/raid/cfl/data/wukong/image_total', image_path+'.jpg')
        if not os.path.exists(image_path1):
            image_path1 = os.path.join('/raid/cfl/data/wukong/image_total', image_path+'.png')
        if not os.path.exists(image_path1):
            image_path1 = os.path.join('/raid/cfl/data/wukong/image_total', image_path+'.jpeg')
        if not os.path.exists(image_path1):
            image_path1 = os.path.join('/raid/cfl/data/wukong/image_total', image_path+'.JPEG')
        if not os.path.exists(image_path1):
            image_path1 = os.path.join('/raid/cfl/data/wukong/image_total', image_path+'.PNG')
        if not os.path.exists(image_path1):
            continue
        try:
            os.system('cp %s /raid/cfl/cn_pretraining_multi_dialog/xllm/xllm/datasets/process_own_dataset/images/' % image_path1)
            print('path: ', image_path)
            print('caption: ', lines[0])
        except:
            continue
    except:
        continue    