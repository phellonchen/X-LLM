import json
import os
from tqdm import tqdm
from PIL import Image
import random

train_path =  '/raid/cfl/data/Chinese-food-net/Chinese_food_net_list_train.json'

image_root = '/raid/cfl/pretraining/en/data/images'
def process(path, split='train'):
    data_total = []
    with open(path, "r") as f_src:
        data = json.load(f_src)
        for item in tqdm(data):
            images = item['image']
            caption = item['caption']
            dataset = item['dataset']
            for image in images:
                info = {
                    'caption': caption,
                    'image': image,
                    'image_id': image,
                    'dataset': dataset
                }
                data_total.append(info)

    return data_total 


train_list = process(train_path)
print('train total: ', len(train_list))
train_save_path = '/raid/cfl/data/Chinese-food-net/Chinese_food_net_train.json'
json.dump(train_list, open(train_save_path, 'w'), indent=4, ensure_ascii=False)



