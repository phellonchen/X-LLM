import json
import os
from tqdm import tqdm

train_file = []
for dirpath, dirnames, filenames in os.walk('/raid/cfl/data/wiki_webtext_baikeqa/wiki_zh_2019/wiki_zh'):
    for filename in filenames:
        if 'wiki' in filename:
            file = os.path.join(dirpath, filename)
            train_file.append(file)


save_path = '/raid/cfl/data/wiki_webtext_baikeqa/wiki_zh_2019/wiki_zh.json'




data_list = []
cnt = 0
image_paths = []
for file_ in train_file:
    cap_data = open(file_)
    for line in tqdm(cap_data.readlines()):
        try:
            line = line.strip()
            lines = json.loads(line)
            ques = lines['title']
            ans = lines['text']
            info = {}
            info['image'] = ''
            info['question'] = ques 
            info['answer'] = ans
            info['dataset'] = 'wiki'
            data_list.append(info)
        except:
            cnt += 1
            print(cnt)

print('total: ', len(data_list))
json.dump(data_list, open(save_path, 'w'), indent=4, ensure_ascii=False)