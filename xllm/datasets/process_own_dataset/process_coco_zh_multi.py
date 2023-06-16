import json
import os
from tqdm import tqdm
caption_txt = '/raid/cfl/en_pretraining/data/images/coco/coco_zh.txt'

save_path = '/raid/cfl/en_pretraining/data/images/coco/coco_zh_multi.json'

cap_data = open(caption_txt)


data_list = []
cnt = 0
image_paths = {}
for line in tqdm(cap_data.readlines()):
    try:
        line = line.strip()
        lines = line.split('\t')
        image_path = lines[0].split('/')[-1].split('-')[0]
        if 'train2014' in image_path:
            image_path = os.path.join('train2014', image_path)
        elif 'val2014' in image_path:
            image_path = os.path.join('val2014', image_path)
        if image_paths.get(image_path) is not None:
            captions = image_paths.get(image_path)
            captions = captions.append(lines[1])
        else:
            image_paths[image_path] = [lines[1]]
    except:
        continue
        
for key, item in image_paths.items():
    caption = item
    info = {}
    info['image'] = key
    info['caption'] = caption
    info['image_id'] = key
    info['dataset'] = 'coco_zh'
    data_list.append(info)


print('total: ', len(data_list))
json.dump(data_list, open(save_path, 'w'), indent=4, ensure_ascii=False)