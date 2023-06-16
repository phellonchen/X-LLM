import json

data = "/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/en_qa90_x-llm_answer1.json"

datas = json.load(open(data))

new_data = []
for item in datas:
    info = {}
    info['question_id'] = item['question_id']
    info['text'] = item['caption_zh']
    new_data.append(info)
print(len(new_data))
json.dump(new_data, open("/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/en_qa90_x-llm_answer.json", 'w'), indent=4, ensure_ascii=False)