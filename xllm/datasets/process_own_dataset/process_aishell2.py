import json
import os
from tqdm import tqdm
from PIL import Image

caption_txt = '/raid/cfl/data/all_audio_cif_feats/aishell2/train_aishell2_zh.json'

save_path = '/raid/cfl/data/all_audio_cif_feats/aishell2/train_aishell2_zh_new.json'
    
cap_data = json.load(open(caption_txt))


data_list = []
cnt = 0
speech_path_json = {}
for line in tqdm(cap_data):
    try:
        speech_path = line['speech_feats_path']
        caption = line['target_text']
        length = line['length']
        utterance_id = line['utterance_id']

        if not os.path.exists(speech_path):
            continue
        if speech_path_json.get(speech_path):
            continue
        else:
            speech_path_json[speech_path] = 1
 
        info = {}
        info['speech_feats_path'] = speech_path
        info['target_text'] = caption
        info['utterance_id'] = utterance_id
        info['length'] = 'length'
        data_list.append(info)
    except:
        cnt += 1
        print(cnt)

print('total: ', len(data_list))
json.dump(data_list, open(save_path, 'w'), indent=4, ensure_ascii=False)
