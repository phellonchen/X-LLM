import json
import os
from tqdm import tqdm
caption_txt = '/raid/cfl/data/dialog/zhdd_lines.txt'

save_path = '/raid/cfl/data/dialog/dialog_dd.json'



cap_data = open(caption_txt)

data_list = []
cnt = 0
image_paths = []

for line in tqdm(cap_data.readlines()):
    try:
        line = line.strip()
        lines = line.split(' ')

        if len(lines) % 2 == 1:
            lines[:-1]

        for idx in range(0, len(lines), 2):
            if idx == 0:
                info = {}
                info['image'] = ''
                info['question'] = "用户：" + lines[idx] + "\\n小元："
                info['answer'] = lines[idx+1] 
                info['dataset'] = 'dailydialog_zh'
                data_list.append(info)
            else:
                history_tmp = lines[:idx]
                template = "用户：{} \\n小元： {}"
                info = {}
                info['image'] = ''
                context = " \\n".join([template.format(history_tmp[i], history_tmp[i+1]) for i in range(0, len(history_tmp), 2)]) + " \\n用户： " + lines[idx]  + " \\n小元："
                if len(list(context)) > 256:
                       continue
                info['question'] = context 

                info['answer'] = lines[idx+1] 
                info['dataset'] = 'dailydialog_zh'
                data_list.append(info)
    except:
        cnt += 1
        print(cnt)

caption_txt = '/raid/cfl/data/dialog/duconv/train.txt'
cap_data = open(caption_txt)
for line in tqdm(cap_data.readlines()):
    try:
        line = line.strip()
        lines = json.loads(line)['conversation']

        if len(lines) % 2 == 1:
            lines[:-1]

        for idx in range(0, len(lines), 2):
            if idx == 0:
                info = {}
                info['image'] = ''
                info['question'] = "用户：" + lines[idx].replace(" ", '') + "\\n小元："
                info['answer'] = lines[idx+1].replace(" ", '')  
                info['dataset'] = 'duconv'
                data_list.append(info)
            else:
                history_tmp = lines[:idx]
                template = "用户：{} \\n小元： {}"
                info = {}
                info['image'] = ''
                context = " \\n".join([template.format(history_tmp[i].replace(" ", ''), history_tmp[i+1].replace(" ", '')) for i in range(0, len(history_tmp), 2)]) + " \\n用户： " + lines[idx].replace(" ", '')  + " \\n小元："
                if len(list(context)) > 256:
                    continue
                info['question'] = context
                info['answer'] = lines[idx+1].replace(" ", '') 
                info['dataset'] = 'duconv'
                data_list.append(info)
    except:
        cnt += 1
        print(cnt)


caption_txt = '/raid/cfl/data/dialog/NaturalConv_Release_20210318/dialog_release.json'
cap_data = open(caption_txt)
for data in cap_data:
    data = json.loads(data.strip())
    for line in tqdm(data):
        try:
            lines = line['content']

            if len(lines) % 2 == 1:
                lines[:-1]

            for idx in range(0, len(lines), 2):
                if idx == 0:
                    info = {}
                    info['image'] = ''
                    info['question'] = "用户：" + lines[idx].replace(" ", '') + "\\n小元："
                    info['answer'] = lines[idx+1].replace(" ", '')  
                    info['dataset'] = 'NaturalConv'
                    data_list.append(info)
                else:
                    history_tmp = lines[:idx]
                    template = "用户：{} \\n小元： {}"
                    info = {}
                    info['image'] = ''
                    context = " \\n".join([template.format(history_tmp[i].replace(" ", ''), history_tmp[i+1].replace(" ", '')) for i in range(0, len(history_tmp), 2)]) + " \\n用户： " + lines[idx].replace(" ", '')  + " \\n小元："
                    if len(list(context)) > 256:
                        continue
                    info['question'] = context
                    info['answer'] = lines[idx+1].replace(" ", '') 
                    info['dataset'] = 'NaturalConv'
                    data_list.append(info)
        except:
            cnt += 1
            print(cnt)

caption_txt = '/raid/cfl/data/dialog/luge_Diamante/luge_Diamante/train.txt'
cap_data = open(caption_txt)
for line in tqdm(cap_data.readlines()):
    try:
        line = line.strip()
        dialog = json.loads(line)['conversation']
        lines = []
        for item in dialog:
            lines.append(item['utterance'])

        if len(lines) % 2 == 1:
            lines[:-1]

        for idx in range(0, len(lines), 2):
            if idx == 0:
                info = {}
                info['image'] = ''
                info['question'] = "用户：" + lines[idx].replace(" ", '') + "\\n小元："
                info['answer'] = lines[idx+1].replace(" ", '')  
                info['dataset'] = 'duDiamante'
                data_list.append(info)
            else:
                history_tmp = lines[:idx]
                template = "用户：{} \\n小元： {}"
                info = {}
                info['image'] = ''
                context = " \\n".join([template.format(history_tmp[i].replace(" ", ''), history_tmp[i+1].replace(" ", '')) for i in range(0, len(history_tmp), 2)]) + " \\n用户： " + lines[idx].replace(" ", '')  + " \\n小元："
                if len(list(context)) > 256:
                    continue
                info['question'] = context
                info['answer'] = lines[idx+1].replace(" ", '') 
                info['dataset'] = 'duDiamante'
                data_list.append(info)
    except:
        cnt += 1
        print(cnt)


caption_txt = '/raid/cfl/data/dialog/DuLeMon/DuLeMon/self/train.json'
cap_data = open(caption_txt)
for line in tqdm(cap_data.readlines()):
    try:
        line = line.strip()
        lines = json.loads(line)['conversation']
        if len(lines) % 2 == 1:
            lines[:-1]

        for idx in range(0, len(lines), 2):
            if idx == 0:
                info = {}
                info['image'] = ''
                info['question'] = "用户：" + lines[idx].replace(" ", '').replace("p1:", "") + "\\n小元："
                info['answer'] = lines[idx+1].replace(" ", '').replace("p2:", "")  
                info['dataset'] = 'dulemon'
                data_list.append(info)
            else:
                history_tmp = lines[:idx]
                template = "用户：{} \\n小元： {}"
                info = {}
                info['image'] = ''
                context = " \\n".join([template.format(history_tmp[i].replace(" ", '').replace("p1:", ""), history_tmp[i+1].replace(" ", '').replace("p2:", "")) for i in range(0, len(history_tmp), 2)]) + " \\n用户： " + lines[idx].replace(" ", '').replace("p1:", "")  + " \\n小元："
                if len(list(context)) > 256:
                    continue
                info['question'] = context
                info['answer'] = lines[idx+1].replace(" ", '').replace("p2:", "") 
                info['dataset'] = 'dulemon'
                data_list.append(info)
    except:
        cnt += 1
        print(cnt)


print('total: ', len(data_list))
json.dump(data_list, open(save_path, 'w'), indent=4, ensure_ascii=False)