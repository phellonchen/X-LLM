import json
import os
from collections import defaultdict

import numpy as np


if __name__ == '__main__':
    base_dir = "/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/tabel"
    review_files = [x for x in os.listdir(base_dir) if x.endswith('.json') and x.startswith('reviwe')]
    review_files = ['/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/xllm/datasets/process_own_dataset/eval_llava_data/tabel/review2.json']
    for review_file in sorted(review_files):
        config = 'X-LLM'
        scores = defaultdict(list)
        print(f'GPT-4 vs. {config}')
        f = json.load(open(review_file))
        for review in f:
            tuple_1 = review['tuple'][0]
            tuple_2 = review['tuple'][1]
            # if tuple_2 == -1:
            #     tuple_2 = 9
            #     tuple_1 = 1
            review['tuple'] = [tuple_2, tuple_1]
            scores[review['category']].append(review['tuple'])
            scores['all'].append(review['tuple'])
        for k, v in scores.items():
            stats = np.asarray(v).mean(0).tolist()
            stats = [round(x, 3) for x in stats]
            print(k, stats, round(stats[1]/stats[0]*100, 1))
        print('=================================')
