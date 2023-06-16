import json
import random

file_path = '/raid/cfl/data/all_audio_cif_feats/aishell2/train_aishell2_zh.json'
save_path = '/raid/cfl/cn_pretraining_multi_dialog/data/lamumo/asr_instruction_zh.json'

datas = json.load(open(file_path))

random.shuffle(datas)

json.dump(datas[:2000], open(save_path, 'w'), indent=4, ensure_ascii=False)