import requests
import json
from tqdm import tqdm
import time
url = 'http://172.18.30.134:4120/chatgpt/'

# response = requests.post(url)

file_path = '/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/qa90_x-llm_answer_400W_from_blip2.json'
# datas_en = json.load(open('/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/en_qa90_x-llm_answer.json'))
# print(len(datas_en))

# translated_index = [_['idx'] for _ in  datas_en]


datas = json.load(open(file_path))
datas_en = []
for idx, data_item in enumerate(tqdm(datas)):
    cnt = 0
    # if str(idx) in translated_index:
    #     continue
    while True:
        try:
            text = data_item['text']
            data = {
                'text': "翻译成英文：" + text
            }
            res = requests.post(url,data=data).json()
            res = res['choices'][0]['message']['content']
            data_item['caption_zh'] = res
            data_item['idx'] = str(idx)
            datas_en.append(data_item)
            json.dump(datas_en, open('/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/en_qa90_x-llm_answer_400W_from_blip2.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
            break
        except:
            if cnt > 10:
                print('Failed for %s' % (cnt, text))
                break
            print('repeat time %d for %s' % (cnt, text))
            cnt += 1
            request_interval = cnt * cnt
            time.sleep(request_interval)
            continue