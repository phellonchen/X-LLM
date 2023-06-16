import torch

image_former_url_or_filename = 'image.pth'
speech_former_url_or_filename = 'speech.pth'
video_former_url_or_filename = 'video.pth'

def get_state_dict(url_or_name):
    checkpoint = torch.load(url_or_name, map_location="cpu")
    if "model" in checkpoint.keys():
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    return state_dict

def get_image_state_dict(url_or_name):
    checkpoint = torch.load(url_or_name, map_location="cpu")
    if "model" in checkpoint.keys():
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    return state_dict

def get_speech_state_dict(url_or_name):
    checkpoint = torch.load(url_or_name, map_location="cpu")
    if "model" in checkpoint.keys():
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint


    return state_dict

def get_video_state_dict(url_or_name):
    checkpoint = torch.load(url_or_name, map_location="cpu")
    if "model" in checkpoint.keys():
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    return state_dict


image_state_dict = get_image_state_dict(image_former_url_or_filename)
speech_image_state_dict = get_speech_state_dict(speech_former_url_or_filename)
video_state_dict = get_video_state_dict(video)

save_to = 'new.pth'
merged_dict = {**speech_image_state_dict,  **image_state_dict, **video_state_dict}
torch.save({'model': merged_dict}, save_to)








