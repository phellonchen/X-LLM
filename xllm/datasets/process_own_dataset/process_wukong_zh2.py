import json
import os
from tqdm import tqdm
from PIL import Image

caption_txt = '/raid/cfl/data/wukong/caption-image-total.txt'

save_path = '/raid/cfl/data/wukong/caption-image-total-134.json'

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
            tmp = Image.open(image_path1)
        except:
            continue
        caption = lines[0]
        info = {}
        info['image'] = image_path1
        info['caption'] = caption
        info['image_id'] = image_path1
        info['dataset'] = 'wukong_zh'
        data_list.append(info)
    except:
        cnt += 1
        print(cnt)

print('total: ', len(data_list))
json.dump(data_list, open(save_path, 'w'), indent=4, ensure_ascii=False)