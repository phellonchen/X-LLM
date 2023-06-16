# @Time    : 2023/6/6
# @Author  : Minglun Han
# @File    : gen_feats_from_service.py
# Bonjour! Wish you good luck, everyday! :)


import os
import sys
import json
import torch
import base64
import requests
import argparse
import urllib
import numpy as np
import torchaudio
import torchaudio.transforms as T


def get_ASRFeature(waveform):
    print("waveform_size: ", waveform.size())
    waveform_bytes = waveform.numpy().tobytes()
    audio_base64 = base64.b64encode(waveform_bytes).decode('utf-8')

    url = 'http://172.18.30.121:8888/genfeats'
    asr_feature = requests.post(url, json={'waveform':audio_base64}).json()

    asr_feature = base64.b64decode(asr_feature)
    asr_feature = np.frombuffer(asr_feature, dtype=np.float32)
    asr_feature = torch.tensor(asr_feature, dtype=torch.float32).view(-1, 512)
    asr_feature = asr_feature.unsqueeze(0)
    print(asr_feature.size())

    return asr_feature


wav_dir = "/raid/cfl/data/all_audio_cif_feats/js_paired_data/wav/js_real_test_data"
text_path = "/raid/cfl/data/all_audio_cif_feats/js_paired_data/wav/js_real_test_data/text"
output_dir = "/raid/cfl/data/all_audio_cif_feats/js_paired_data/"
output_feats_dir = "/raid/cfl/data/all_audio_cif_feats/js_paired_data/dumped_cif_feats/"
output_json_fn = "js_real_test_data.json"


# load text
id2text = dict()
with open(text_path, "r") as f_text:
    for line in f_text:
        utt_id, text = line.strip().split("\t")
        utt_id = utt_id.split(".")[0]
        id2text[utt_id] = text.strip()


n = 0
json_save_list = []
fn_list = os.listdir(wav_dir)
for fn in fn_list:
    if ".wav" not in fn:
        continue
    utt_id = fn.strip().split(".")[0]
    wav_path = os.path.join(wav_dir, fn)

    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        resampler = T.Resample(sr, 16000, dtype=waveform.dtype)
        waveform = resampler(waveform)
    cif_feats = get_ASRFeature(waveform=waveform)[0]   # T x C
    text = id2text[utt_id]
    text = text.replace(".", "")
    text = text.replace("。", "")
    text = text.replace("？", "")
    text = text.replace("，", "")
    text = text.replace(",", "")
    text = text.replace("：", "")

    # save feats
    utt_id = "js_" + utt_id
    cur_save_dir = os.path.join(output_feats_dir, utt_id + ".npy")
    length = cif_feats.size(0)
    np.save(cur_save_dir, cif_feats[:length])

    json_save_list.append(
        {
            "utterance_id": utt_id,
            "speech_feats_path": cur_save_dir,
            "length": length,
            "wav_length": waveform.size()[-1],
            "target_text": text,
        }
    )

    n += 1
    # if n == 10:
    #     break
    if n % 1000 == 0:
        print("have processed %d samples" % n )


json_str = json.dumps(
    json_save_list, indent=4, ensure_ascii=False
)
output_fn = os.path.join(output_dir, output_json_fn)
with open(output_fn, "w") as f_out:
    f_out.write(json_str)
