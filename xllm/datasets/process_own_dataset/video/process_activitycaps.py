import json

# file_path = '/raid/cfl/data/video_caption/activitycaps/train.json'

# datas = json.load(open(file_path, 'r'))

# new_data = []
# for video, data in datas.items():
#     duration = data['duration']
#     if duration > 60:
#         continue
#     info = {}
#     info['video'] = video
#     info['caption'] = " ".join(data['sentences'])
#     info['sentences'] = "\n".join(data['sentences'])
#     info['duration'] = duration
#     new_data.append(info)
# print(len(new_data))

# json.dump(new_data, open('/raid/cfl/data/video_caption/activitycaps/train_new.json', 'w'), indent=4)

file_path = '/raid/cfl/data/video_caption/activitycaps/train_new_fliter_new_chatgpt.json'

datas = json.load(open(file_path, 'r'))['annotations']


print(len(datas))

json.dump(datas, open('/raid/cfl/data/video_caption/activitycaps/train_fliter_chatgpt_20230422.json', 'w'), indent=4, ensure_ascii=False)