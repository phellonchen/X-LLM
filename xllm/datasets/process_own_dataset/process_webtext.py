import json
import os
from tqdm import tqdm
caption_txt = '/raid/cfl/data/wiki_webtext_baikeqa/webtext2019zh/web_text_zh_train.json'

save_path = '/raid/cfl/data/wiki_webtext_baikeqa/webtext2019zh/web_text_zh_train_zh.json'

cap_data = open(caption_txt)


data_list = []
cnt = 0
image_paths = []
for line in tqdm(cap_data.readlines()):
    try:
        line = line.strip()
        lines = json.loads(line)
        ques = lines['title']
        desc = lines['desc']
        ans = lines['content']
        info = {}
        info['image'] = ''
        info['question'] = ques + ' 问题描述：' + desc
        info['answer'] = ans
        info['dataset'] = 'web_text'
        data_list.append(info)
    except:
        cnt += 1
        print(cnt)

print('total: ', len(data_list))
json.dump(data_list, open(save_path, 'w'), indent=4, ensure_ascii=False)