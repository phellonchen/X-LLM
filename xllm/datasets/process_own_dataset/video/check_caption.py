import pickle
import json
data_file = open("/raid/cfl/data/video_caption/MSRVTT/annotation/MSR_VTT.json")
data = json.load(data_file)
print('sss')

new_data = []
video_dict = {}
for item in data['annotations']:
    if video_dict.get(item['image_id']):
        continue
    new_data.append(item)
    video_dict[item['image_id']] = 1

print('len: ', len(new_data))
json.dump(new_data, open("/raid/cfl/data/video_caption/MSRVTT/annotation/MSR_VTT_new.json", 'w'), indent=4)