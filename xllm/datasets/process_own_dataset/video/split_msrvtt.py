import json

all_data = json.load(open("/raid/cfl/data/video_caption/MSRVTT/annotation/MSR_VTT_new_ali_zh.json"))

train_list = {}
test_list = {}

train_data = []
test_data = []

for line in open("/raid/cfl/data/video_caption/MSRVTT/videos/train_list_new.txt").readlines():
    line = line.strip()
    if train_list.get(line):
        continue
    train_list[str(line)] = 1

for line in open("/raid/cfl/data/video_caption/MSRVTT/videos/test_list_new.txt").readlines():
    line = line.strip()
    if test_list.get(line):
        continue
    test_list[str(line)] = 1


for item in all_data:
    image_id = item['image_id']
    if train_list.get(image_id):
        train_data.append(item)
    elif test_list.get(image_id):
        test_data.append(item)

print('train: ', len(train_data))
print('test: ', len(test_data))
json.dump(train_data, open("/raid/cfl/data/video_caption/MSRVTT/annotation/MSR_VTT_new_ali_zh_train.json", 'w'), indent=4, ensure_ascii=False)
json.dump(test_data, open("/raid/cfl/data/video_caption/MSRVTT/annotation/MSR_VTT_new_ali_zh_test.json", 'w'), indent=4, ensure_ascii=False)