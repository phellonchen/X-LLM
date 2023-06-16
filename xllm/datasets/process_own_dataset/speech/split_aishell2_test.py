import numpy as np
import csv
import os
import json

split = 'mic'
file_name = "/raid/cfl/data/all_audio_cif_feats/165_org_data/test-%s-aishell2.npy" % split

datas = np.load(open(file_name, 'rb'), allow_pickle=True)
print('ss')

csv_file = "/raid/cfl/data/all_audio_cif_feats/org_data_index/test-%s-aishell2.tsv" % split
csv_data = open(csv_file)
text_dict = {}
for idx, row in enumerate(csv_data.readlines()):
    if idx == 0:
        continue
    row = row.strip().split('\t')
    text_dict[row[0]] = row[3]

llm_speech_data = []

for data in datas:
    speech_id = data[0]
    speech_features = data[1]

    speech_text = text_dict.get(speech_id)

    path = "/raid/cfl/data/all_audio_cif_feats/split_shards/test-%s-aishell2" % split
    path = os.path.join(path, speech_id+'.npy')
    np.save(open(path, 'wb'), speech_features)

    info = {}
    info["utterance_id"] = str(speech_id)
    info["speech_feats_path"] = path
    info['length'] = str(data[4])
    info['target_text'] = str(speech_text).replace(" ", "")

    llm_speech_data.append(info)

json.dump(llm_speech_data, open('/raid/cfl/data/all_audio_cif_feats/aishell2/test-%s-aishell2_zh.json' % split, 'w'), indent=4, ensure_ascii=False)


