import json
from xllm.common.rawvideo_util import RawVideoExtractor
import numpy as np
import os
from tqdm import tqdm
feature_framerate = 1
image_resolution = 224
max_frames = 5
slice_framepos = 0
frame_order = 0
rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
vis_root = '/raid/cfl/data/video_caption/activitycaps/video_all_compressed'

def _get_rawvideo(item, s=[0], e=[60]):
    video_path = os.path.join(vis_root, item['video'] + ".mp4")
    try:
        for i in range(len(s)):
            start_time = 0
            end_time = int(item['duration'])
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = end_time + 1

            # Should be optimized by gathering all asking of this video
            raw_video_data = rawVideoExtractor.get_video_data(video_path, start_time, end_time)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                return True
    except:
        pass
    return False


file_path = '/raid/cfl/data/video_caption/activitycaps/train_new_fliter.json'

datas = json.load(open(file_path, 'r'))

new_datas = []
for item in tqdm(datas, total=len(datas)):
    if _get_rawvideo(item):
        new_datas.append(item)

print(len(new_datas))

json.dump(new_datas, open('/raid/cfl/data/video_caption/activitycaps/train_new_fliter_new.json', 'w'), indent=4, ensure_ascii=False)