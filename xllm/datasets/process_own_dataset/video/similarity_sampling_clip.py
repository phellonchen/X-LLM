import torch
from xllm.models.eva_vit import create_eva_vit_g
import cv2
from PIL import Image
from xllm.models.clip import build_model
img_size = 224
drop_path_rate=0 
use_grad_checkpoint = False
precision = "fp16"


def get_clip_video_frames(video_path, image_process):
    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    sample_time = FPS // 3
    imgs = []

    i = 0
    while (cap.isOpened()):
        ret, cv2_im = cap.read()

        if ret and i % sample_time == 0:
            converted = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(converted)
            imgs.append(pil_im)
        elif not ret:
            break

        i += 1

    cap.release()

    images = torch.cat([image_process(x).unsqueeze(0) for x in imgs])

    return images

def filter_video(image_fts, similiarities):
    THRESHOLD = 0.9
    groups = []
    curr_group = []
    for i in range(similiarities.size(0)):
        if len(curr_group) == 0:
            curr_group.append(i)

        if i + 1 == similiarities.size(0):
            if len(curr_group) >= 1:
                groups.append(curr_group)
            break

        if similiarities[curr_group[0]][i + 1] > THRESHOLD:
            curr_group.append(i + 1)
        else:
            if len(curr_group) >= 1:
                groups.append(curr_group)
            curr_group = []

    result_features = []
    selected_indices = []
    if len(groups) >= 1:
        for i, group in enumerate(groups):
            result_features.append(image_fts[group[0]])
            selected_indices.append(group[0])

    return torch.stack(result_features), selected_indices

def run_video(video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
   
    visual_encoder, clip_preprocess = build_model("ViT-B/32")
    visual_encoder = visual_encoder.to(device)

    video_frames = get_clip_video_frames(video_path, clip_preprocess).to(device)

    with torch.no_grad():
        frames_fts = visual_encoder(video_frames).detach()
        frames_fts = torch.nn.functional.normalize(frames_fts, dim=-1).detach()
        similiarities = frames_fts @ frames_fts.T
        image_fts, selected_frames_indices = filter_video(frames_fts, similiarities)

    print('ssss')


video_file = '/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/v_juKQ_gU42EM.mp4'

run_video(video_file)