import json
import os
from tqdm import tqdm
from PIL import Image
import random
import copy
train_path =  '/raid/cfl/cn_dialog/data/cn_data/cn_visdial_1.0_train_data_cn.json'
val_path =  '/raid/cfl/cn_dialog/data/cn_data/cn_visdial_1.0_val_data_cn.json'
# train_path =  '/raid/cfl/en_pretraining/data/images/cn_mbart_50_visdial_1.0_train.json'
# val_path =  '/raid/cfl/en_pretraining/data/images/cn_mbart_50_visdial_1.0_val.json'
image_root = '/raid/cfl/pretraining/en/data/images'
def process(path, split='train'):
    data_total = []
    example_type = 'visdial'
    # prompt = ['你能看见什么？','请描述一下这张图片','这幅图片的内容是什么？','这张图片上有什么','从这张图片上你能获得什么信息？', '图里有什么', '这是什么图片','图片描绘了什么？', '用语言描述一下这幅图片']
    prompt = '请描述一下这张图片'
    with open(path, "r") as f_src:
        data = json.load(f_src)['data']
        counter = 0
        dialogs = data['dialogs']
        questions_list = data['questions']
        answers_list = data['answers']
        for dialog in tqdm(dialogs):
            img_id = dialog['image_id']
            caption = dialog['caption'].strip() 
            
            ques_id = [item['question'] for item in dialog['dialog']]
            ans_id = [item['answer'] for item in dialog['dialog']]
            ans_opts = [item['answer_options'] for item in dialog['dialog']]
            ans_opts_tokens = [[answers_list[id] for id in ans] for ans in ans_opts]
            gt_id = [item['gt_index'] for item in dialog['dialog']]
            questions= [questions_list[id]  for id in ques_id]
            answers = [answers_list[id] for id in ans_id]
            if split == 'val':
                img_pth = os.path.join(image_root, 'VisualDialog_val2018', 'VisualDialog_val2018_%012d.jpg' % img_id)
            else:
                img_pth = os.path.join(image_root,'train2014', 'COCO_train2014_%012d.jpg' % img_id)
                if not os.path.exists(img_pth):
                    img_pth = os.path.join(image_root, 'val2014','COCO_val2014_%012d.jpg' % img_id)
    

            # separator = '  |!!!|  '
            input_text = "用户：" + prompt+ " \\n小元："
            # data_total.append(
            #     {'image': img_pth,
            #         'question':input_text,
            #         'answer': caption,
            #         'image_id': img_id,
            #         'dataset': 'visdial_zh'}
            # )
            history = []
            # history.append((prompt, caption))
            template = "用户：{} \\n小元： {}"
            for turn_i in range(10):
                if turn_i == 0:
                    history_tmp = []
                    history_tmp.append((prompt, caption))
                    input_text = " \\n".join([template.format(history_tmp[i][0], history_tmp[i][1]) for i in range(len(history_tmp))]) + " \\n用户： " + questions[turn_i].replace("?", "？").strip()  + " \\n小元："
                # elif turn_i > 3:
                #     history_tmp = copy.deepcopy(history[-3:])
                #     history_tmp.insert(0, (prompt, caption))
                #     input_text = " \\n".join([template.format(history_tmp[i][0], history_tmp[i][1]) for i in range(len(history_tmp))]) + " \\n用户： " + questions[turn_i].replace("?", "？").strip()  + " \\n小元："
                else:
                    history_tmp = copy.deepcopy(history)
                    history_tmp.insert(0, (prompt, caption))
                    input_text = " \\n".join([template.format(history_tmp[i][0], history_tmp[i][1]) for i in range(len(history_tmp))]) + " \\n用户： " + questions[turn_i].replace("?", "？").strip()  + " \\n小元："

                input_text = input_text.replace('\t', '')
                answer = answers[turn_i].replace('\t', '').strip() 
                if len(list(input_text)) > 250:
                    continue
                history.append((questions[turn_i].replace("?", "？").strip(), answer))
                
                data_total.append(
                        {'image': img_pth,
                         'question':input_text,
                         'answer': answer,
                         'image_id': img_id,
                         'dataset': 'visdial_zh'}
                    )

    return data_total 


train_list = process(train_path)
print('train total: ', len(train_list))
train_save_path = '/raid/cfl/en_pretraining/data/images/vd/vd_zh_train_4turns_nocaptioning.json'
json.dump(train_list, open(train_save_path, 'w'), indent=4, ensure_ascii=False)
val_list = process(val_path, split='val')
print('val total: ', len(val_list))
val_save_path = '/raid/cfl/en_pretraining/data/images/vd/vd_zh_val_4turns_nocaptioning.json'
json.dump(val_list, open(val_save_path, 'w'), indent=4, ensure_ascii=False)


