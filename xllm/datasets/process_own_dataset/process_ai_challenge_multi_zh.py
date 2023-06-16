import json
import os
from tqdm import tqdm
from PIL import Image

caption_txt = ['/raid/cfl/data/ai_challenge/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json',
               '/raid/cfl/data/ai_challenge/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json',
            #    '/raid/cfl/data/ai_challenge/ai_challenger_caption_test_a_20180103/caption_test_a_annotations_20180103.json',
               ]

save_path = '/raid/cfl/data/ai_challenge/ai_challenge_multi.json'




data_list = []
cnt = 0

for file in caption_txt:
    cap_data = json.load(open(file))
    for item in tqdm(cap_data):
        try:           
            image_path = item["image_id"]
            if 'caption_test_a' in file:
                image_path = os.path.join('caption_test_a_images_20180103', image_path)
            elif 'caption_validation' in file:
                image_path = os.path.join('caption_validation_images_20170910', image_path)
            else:
                image_path = os.path.join('caption_train_images_20170902', image_path)

            if not os.path.exists(os.path.join('/raid/cfl/data/ai_challenge', image_path)):
                continue
            try:
                image_ = Image.open(os.path.join('/raid/cfl/data/ai_challenge', image_path))
            except:
                continue
            caption = item['caption']
            info = {}
            info['image'] = image_path
            info['caption'] = caption
            info['image_id'] = image_path
            info['dataset'] = 'ai_challenge_zh'
            data_list.append(info)
        except:
            continue


print('total: ', len(data_list))
json.dump(data_list, open(save_path, 'w'), indent=4, ensure_ascii=False)