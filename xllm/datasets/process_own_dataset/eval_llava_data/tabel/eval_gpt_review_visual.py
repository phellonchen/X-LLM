import argparse
import json
import os
import requests
# import openai
from tqdm import tqdm
# import ray
import time

url = 'http://172.18.30.134:4121/chatgpt/'
def get_eval(content: str, max_tokens: int):
    while True:
        try:
            data = {
                'text': content
            }
            res = requests.post(url,data=data).json()
            res = res['choices'][0]['message']['content']
            break
        except Exception as e:
            print(e)
        
        time.sleep(5)

    print('success!')
    return res

# @ray.remote(num_cpus=4)
# def get_eval(content: str, max_tokens: int):
#     while True:
#         try:
#             response = openai.ChatCompletion.create(
#                 model='gpt-4',
#                 messages=[{
#                     'role': 'system',
#                     'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
#                 }, {
#                     'role': 'user',
#                     'content': content,
#                 }],
#                 temperature=0.2,  # TODO: figure out which temperature is best for evaluation
#                 max_tokens=max_tokens,
#             )
#             break
#         except openai.error.RateLimitError:
#             pass
#         except Exception as e:
#             print(e)
#         time.sleep(1)

#     print('success!')
#     return response['choices'][0]['message']['content']


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question', default='/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/tabel/question90.jsonl')
    parser.add_argument('-c', '--context', default='/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/tabel/caps_boxes_coco2014_val_80.jsonl')
    parser.add_argument('-a', '--answer-list', nargs='+', default=['/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/tabel/qa90_x-llm_answer.jsonl','/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/tabel/qa90_gpt4_answer.jsonl'])
    parser.add_argument('-r', '--rule', default='/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/tabel/rule.json')
    parser.add_argument('-o', '--output', default='/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/tabel/review2.json')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    # ray.init()

    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    review_file = open(f'{args.output}', 'w')

    context_list = [json.loads(line) for line in open(os.path.expanduser(args.context))]
    image_to_context = {context['image']: context for context in context_list}

    js_list = []
    handles = []
    idx = 0
    for ques_js, ans1_js, ans2_js in tqdm(zip(f_q, f_ans1, f_ans2)):
        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)

        inst = image_to_context[ques['image']]
        cap_str = '\n'.join(inst['captions'])
        box_str = '\n'.join([f'{instance["category"]}: {instance["bbox"]}' for instance in inst['instances']])

        category = json.loads(ques_js)['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            assert False, f"Visual QA category not found in rule file: {category}."
        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Context]\n{cap_str}\n\n{box_str}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        js_list.append({
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ans2.get('answer_id', ans2['question_id']),
            'category': category})
        idx += 1
        handles.append(get_eval(content, args.max_tokens))
        # To avoid the rate limit set by OpenAI
        time.sleep(1)

    # reviews = ray.get(handles)
    reviews = handles
    res = []
    for idx, review in enumerate(reviews):
        scores = parse_score(review)
        js_list[idx]['content'] = review
        js_list[idx]['tuple'] = scores
        # review_file.write(json.dumps(js_list[idx]) + '\n')
        res.append(js_list[idx])
    # review_file.close()
    json.dump(js_list, review_file, indent=4, ensure_ascii=False)