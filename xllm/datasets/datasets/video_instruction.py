"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from xllm.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import random
import numpy as np
from xllm.common.rawvideo_util import RawVideoExtractor

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": os.path.basename(ann["image"]),
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class VideoInstructionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.prompt = [
            '请描述这个视频。',
            '请查看这个视频并描述您注意到的内容。',
            '请提供这个视频的描述',
            '你能为我描述这个视频的内容吗？',
        ]
        feature_framerate = 1
        image_resolution = 224
        self.max_frames = 5
        self.slice_framepos = 0
        self.frame_order = 0
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index] 


        caption = self.text_processor(ann["caption_zh"])
        # text_input = '[gMASK]'
        text_input = '问：' + random.choice(self.prompt) + '\n答：'
        video = self._get_rawvideo(index)

        return {"video": video[0].squeeze(2).squeeze(0), "text_input": text_input, "text_output": caption, "task_id": 5,}
      
    def _get_rawvideo(self, idx, s=[0], e=[60]):
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.int32)
        max_video_length = [0] * len(s)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(s), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float32)
        video_path = os.path.join(self.vis_root, self.annotation[idx]['image_id'] + ".mp4")
        try:
            for i in range(len(s)):
                start_time = 0
                # end_time = int(self.annotation[idx]['duration'])
                end_time = 60
                start_time = start_time if start_time >= 0. else 0.
                end_time = end_time if end_time >= 0. else 0.
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = end_time + 1

                # Should be optimized by gathering all asking of this video
                raw_video_data = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time)
                raw_video_data = raw_video_data['video']

                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    # L x T x 3 x H x W
                    raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                    if self.max_frames < raw_video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = raw_video_slice[:self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = raw_video_slice[-self.max_frames:, ...]
                        else:
                            sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                            video_slice = raw_video_slice[sample_indx, ...]
                    else:
                        video_slice = raw_video_slice

                    video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                    slice_len = video_slice.shape[0]
                    max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                    if slice_len < 1:
                        pass
                    else:
                        video[i][:slice_len, ...] = video_slice
                else:
                    print("video path: {} error. video id: {}, start: {}, end: {}".format(video_path, idx, start_time, end_time))
        except Exception as excep:
            print("video path: {} error. video id: {}, start: {}, end: {}, Error: {}".format(video_path, idx, s, e, excep))
            raise excep

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

