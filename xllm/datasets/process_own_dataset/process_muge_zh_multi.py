import json
import os
from tqdm import tqdm
caption_txt = ['/raid/cfl/data/MUGE/train_texts.jsonl',
               ]

save_path = '/raid/cfl/data/MUGE/muge_zh_multi.json'



data_list = []
for file in caption_txt:
    cap_data = open(file)

    cnt = 0
    image_paths = {}
    for line in tqdm(cap_data.readlines()):
        try:
            line = line.strip()
            lines = json.loads(line)
            image_path = str(lines['image_ids'][0])

            if image_paths.get(image_path) is not None:
                captions = image_paths.get(image_path)
                captions = captions.append(lines['text'])
            else:
                image_paths[image_path] = [lines['text']]
        except:
            continue
            
    if 'train' in file:
        image_ = '/raid/cfl/data/MUGE/train_imgs.tsv'
    else:
        image_ = '/raid/cfl/data/MUGE/valid_imgs.tsv'
    image_ = open(image_)
    image_feature = {}
    for line in image_.readlines():
        line = line.strip()
        line = line.split('\t')
        image_feature[line[0]] = line[1]

    for key, item in image_paths.items():
        caption = item
        info = {}
        info['image'] = image_feature[key]
        info['caption'] = caption
        info['image_id'] = image_path
        info['dataset'] = 'MUGE_zh'
        data_list.append(info)


print('total: ', len(data_list))
json.dump(data_list, open(save_path, 'w'), indent=4, ensure_ascii=False)