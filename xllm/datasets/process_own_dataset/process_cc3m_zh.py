import json
import os
from tqdm import tqdm
from PIL import Image, TiffImagePlugin
import piexif

caption_txt = '/raid/cfl/en_pretraining/data/images/cc3m/cc3m_zh.txt'

save_path = '/raid/cfl/en_pretraining/data/images/cc3m/cc3m_zh_new.json'

cap_data = open(caption_txt)


data_list = []
cnt = 0
image_paths = []
for line in tqdm(cap_data.readlines()):
    try:
        line = line.strip()
        lines = line.split('\t')
        image_path = lines[0]

   
        image_path = os.path.join('/raid/cfl/en_pretraining/data/images/cc3m/', 'training', image_path)
        if not os.path.exists(image_path):
            image_path = os.path.join('/raid/cfl/en_pretraining/data/images/cc3m/', 'validation', image_path)  
        if not os.path.exists(image_path):
            continue
        try:
            tmp = Image.open(image_path)
   
            width, height = tmp.size
            if min(width, height) < 224 or max(width/height, height/width) > 3:
                continue
            if width * height > 89478485:
                ratio = width / height
                new_width = int((89478485 / ratio) ** 0.5)
                new_height = int(new_width / ratio)
                
                # Resize the image to the new size
                tmp = tmp.resize((new_width, new_height))
                
                # Save the resized image
                tmp.save(image_path)

        except:
            continue
        caption = lines[1]
        info = {}
        info['image'] = image_path
        info['caption'] = caption
        info['image_id'] = image_path
        info['dataset'] = 'cc3m_zh'
        data_list.append(info)
    except:
        cnt += 1
        print(cnt)

print('total: ', len(data_list))
json.dump(data_list, open(save_path, 'w'), indent=4, ensure_ascii=False)
