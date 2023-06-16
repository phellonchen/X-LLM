import json
import os
from tqdm import tqdm
caption_txt = '/raid/cfl/data/pclue/pCLUE_train.json'

save_path = '/raid/cfl/data/pclue/pCLUE_train_zh.json'

cap_data = open(caption_txt)


data_list = []
cnt = 0
image_paths = []
for line in tqdm(cap_data.readlines()):
    try:
        line = line.strip()
        lines = json.loads(line)
        ques = lines['input'].replace('\n答案：', '')
        ans = lines['target']
        ques_type = lines['type']
        info = {}
        info['image'] = ''
        info['question'] = ques
        info['answer'] = ans
        info['dataset'] = 'pCLUE_' + ques_type
        data_list.append(info)
    except:
        cnt += 1
        print(cnt)

print('total: ', len(data_list))
json.dump(data_list, open(save_path, 'w'), indent=4, ensure_ascii=False)