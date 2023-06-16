import json
import os
from tqdm import tqdm
from PIL import Image

caption_txt = '/raid/cfl/cn_pretraining_multi_dialog/data/visdial_cn/train_vsdial_asr_zh.json'

save_path = '/raid/cfl/cn_pretraining_multi_dialog/data/visdial_cn/train_vsdial_asr_zh_new.json'
    
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
        speech_path = speech_path.split('/')[-1]
        speech_path = os.path.join('/raid/cfl/cn_pretraining_multi_dialog/data/visdial_cn/vsdial_cn_cif_feats/shards/train', speech_path)
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
