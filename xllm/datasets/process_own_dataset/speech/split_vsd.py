import numpy as np
import csv
import os
import json

split = 'v3'
file_name = "/raid/cfl/data/all_audio_cif_feats/165_org_data/test-%s-vsdial.npy" % split

datas = np.load(open(file_name, 'rb'), allow_pickle=True)
print('ss')
if split == 'v2':
    csv_file = "/raid/cfl/data/all_audio_cif_feats/165_org_data/test_v2_2k.tsv"
else:
    csv_file = "/raid/cfl/data/all_audio_cif_feats/165_org_data/test_v3_300_202210.tsv"
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

    path = "/raid/cfl/data/all_audio_cif_feats/split_shards/test-%s-vsd" % split
    path = os.path.join(path, speech_id+'.npy')
    np.save(open(path, 'wb'), speech_features)

    info = {}
    info["utterance_id"] = str(speech_id)
    info["speech_feats_path"] = path
    info['length'] = str(data[4])
    info['target_text'] = str(speech_text).replace(" ", "")
    info['image'] = "/raid/cfl/pretraining/en/data/images/VisualDialog_test2018_%012d.jpg" % int(speech_id.split('_')[0])

    llm_speech_data.append(info)

json.dump(llm_speech_data, open('/raid/cfl/data/all_audio_cif_feats/aishell2/test-%s-visdial_zh.json' % split, 'w'), indent=4, ensure_ascii=False)


