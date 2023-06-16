import json
import os
from tqdm import tqdm
from PIL import Image

caption_txt = '/raid/cfl/data/all_audio_cif_feats/visdial_cn/train_vsdial_zh_nocap.json'

save_path = '/raid/cfl/data/all_audio_cif_feats/visdial_cn/train_vsdial_zh_nocap_new.json'
    
cap_data = json.load(open(caption_txt))


data_list = []
cnt = 0
speech_path_json = {}
for line in tqdm(cap_data):
    try:
        speech_path = line['speech_feats_path']
        if not os.path.exists(speech_path):
            continue

    
        data_list.append(line)
    except:
        cnt += 1
        print(cnt)

print('total: ', len(data_list))
json.dump(data_list, open(save_path, 'w'), indent=4, ensure_ascii=False)
