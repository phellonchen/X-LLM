import json
import random

file_path = '/raid/cfl/cn_pretraining_multi_dialog/data/lamumo/chatgpt_filter_cap_zh_check.json'
save_path = '/raid/cfl/cn_pretraining_multi_dialog/data/lamumo/chatgpt_filter_cap_zh_check_new.json'

datas = json.load(open(file_path))['annotations']


json.dump(datas, open(save_path, 'w'), indent=4, ensure_ascii=False)