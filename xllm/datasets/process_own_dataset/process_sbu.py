import json
import os
from tqdm import tqdm
from PIL import Image


save_path = '/raid/cfl/en_pretraining/data/images/sbu/sbu_zh_new.json'

train_file = []
for dirpath, dirnames, filenames in os.walk('/raid/cfl/data/sbu-mm-text-only'):
    for filename in filenames:
        if '.data' in filename:
            file = os.path.join(dirpath, filename)
            train_file.append(file)


sub_data = json.load(open('/raid/cfl/en_pretraining/data/images/sbu/sbu.json'))

sub_json = {}
for item in sub_data:
    sub_json[item['caption']] = item['image']


data_list = []
cnt = 0
image_paths = []
for file in tqdm(train_file):
    data = open(file, 'r')
    for line in data.readlines():
        line = json.loads(line.strip())
        caption = line['caption']['zh']
        caption_en = line['caption']['en']
        image = sub_json.get(caption_en)

        image_path = image.split('/')[-1].split('_')[-1]

        image_path1 = os.path.join('/raid/cfl/en_pretraining/data/images/sbu/pythonDownload/subpic', image_path)
        if not os.path.exists(image_path1):
            continue
        try:
            image_ = Image.open(image_path1)
            width, height = image_.size
            if min(width, height) < 224 or max(width/height, height/width) > 3:
                continue
        except:
            continue
    
        info = {}
        info['image'] = image_path1
        info['caption'] = caption
        info['image_id'] = image_path
        info['dataset'] = 'sbu_zh'
        data_list.append(info)


print('total: ', len(data_list))
json.dump(data_list, open(save_path, 'w'), indent=4, ensure_ascii=False)
