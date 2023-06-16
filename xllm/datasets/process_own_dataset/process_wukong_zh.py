import json
import os
from tqdm import tqdm
from PIL import Image

caption_txt = '/raid/cfl/data/wukong/caption-image-total.json'

save_path = '/raid/cfl/data/wukong/caption-image-total-3000w.json'
    
cap_data = json.load(open(caption_txt))


data_list = []
cnt = 0
image_paths = []
for line in tqdm(cap_data):
    try:
        image_path = line['image']
        caption = line['caption']

        image_path = os.path.join('/raid/cfl/data/wukong/image_total', image_path + '.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join('/raid/cfl/data/wukong/image_total', image_path + '.png')  
        if not os.path.exists(image_path):
            continue
 
        info = {}
        info['image'] = image_path
        info['caption'] = caption
        info['image_id'] = image_path
        info['dataset'] = 'wukong_zh'
        data_list.append(info)
    except:
        cnt += 1
        print(cnt)

print('total: ', len(data_list))
json.dump(data_list, open(save_path, 'w'), indent=4, ensure_ascii=False)
