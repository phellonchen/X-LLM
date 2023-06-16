import json
import os
from tqdm import tqdm
caption_txt = '/raid/cfl/en_pretraining/data/images/coco/coco_zh.txt'

save_path = '/raid/cfl/en_pretraining/data/images/coco/coco_zh.json'

cap_data = open(caption_txt)


data_list = []
cnt = 0
image_paths = []
for line in tqdm(cap_data.readlines()):
    try:
        line = line.strip()
        lines = line.split('\t')
        image_path = lines[0].split('/')[-1].split('-')[0]
        if 'train2014' in image_path:
            image_path = os.path.join('train2014', image_path)
        elif 'val2014' in image_path:
            image_path = os.path.join('val2014', image_path)
        if image_path in image_paths:
            continue
        else:
            image_paths.append(image_path)
        caption = lines[1]
        info = {}
        info['image'] = image_path
        info['caption'] = caption
        info['image_id'] = image_path
        info['dataset'] = 'coco_zh'
        data_list.append(info)
    except:
        cnt += 1
        print(cnt)

print('total: ', len(data_list))
json.dump(data_list, open(save_path, 'w'), indent=4, ensure_ascii=False)