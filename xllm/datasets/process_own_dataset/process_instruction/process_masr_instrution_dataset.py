import json
import random

file_path = '/raid/cfl/data/all_audio_cif_feats/visdial_cn/train_vsdial_zh_nocap_new.json'
save_path = '/raid/cfl/cn_pretraining_multi_dialog/data/lamumo/masr_instruction_zh.json'

datas = json.load(open(file_path))

random.shuffle(datas)

json.dump(datas[:2000], open(save_path, 'w'), indent=4, ensure_ascii=False)