import json

data = "/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/en_qa90_x-llm_answer_400W_from_blip2.json"

datas = json.load(open(data))

new_data = []
for item in datas:
    info = {}
    info['question_id'] = item['question_id']
    info['text'] = item['caption_zh']
    new_data.append(info)
print(len(new_data))

output = open('/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/tabel/qa90_x-llm_answer_400W_from_blip2.jsonl', 'w')

for item in new_data:
    output.write(json.dumps(item) + '\n')

output.close()
