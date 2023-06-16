import streamlit as st
from PIL import Image
import torch
from xllm.models import load_model_and_preprocess
from xllm.models import xllm_models

import numpy as np
import cv2
from streamlit_chat import message
from fastapi import FastAPI
import uvicorn
import requests
import json
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import os
from streamlit_option_menu import option_menu
import time
import jieba
import jieba.posseg as pseg
from xllm.common.rawvideo_util import RawVideoExtractor
import tempfile
import base64
import torchaudio

STREAMLIT_RUNNING_PATH="demo/audio"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Streamlit App
st.set_page_config(
    page_title="Streamlit Chat - Demo",
    page_icon=":robot:"
)
# app = FastAPI()
def recorder_factory():
    return MediaRecorder(f"{STREAMLIT_RUNNING_PATH}/latest_record.wav")
# asr_json = json.load(open('/raid/cfl/cn_pretraining_multi_dialog/xllm/demo/vd_speech_wav2vec.json', encoding='utf-8'))

def get_ASRFeature(speech):
    print(speech)
    waveform, samplereate = torchaudio.load(speech)
    waveform_bytes = waveform.numpy().tobytes()
    url = 'http://172.18.30.121:8888/genfeats'
    audio_base64 = base64.b64encode(waveform_bytes).decode('utf-8')
    asr_feature = requests.post(url, json={'waveform':audio_base64}).json()
    # print(asr_text)
    return asr_feature


@st.cache_resource
def get_VideoExtractor():
    rawVideoExtractor = RawVideoExtractor(framerate=1, size=224)
    return rawVideoExtractor
rawVideoExtractor = get_VideoExtractor()

def process_video(video_path):
    max_frames = 10
    slice_framepos = 0
    frame_order = 0
    max_video_length = [0] * 1
    # video_path = "/raid/cfl/cn_pretraining_multi_dialog/xllm-speech-lora/demo/video/upload_video.mp4"
    # Pair x L x T x 3 x H x W
    video = np.zeros((1, max_frames, 1, 3,
                        rawVideoExtractor.size, rawVideoExtractor.size), dtype=np.float32)
    video_path = video_path
    try:
        for i in range(1):
            # Should be optimized by gathering all asking of this video
            raw_video_data = rawVideoExtractor.get_video_data(video_path, start_time=0, end_time=60)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if max_frames < raw_video_slice.shape[0]:
                    if slice_framepos == 0:
                        video_slice = raw_video_slice[:max_frames, ...]
                    elif slice_framepos == 1:
                        video_slice = raw_video_slice[-max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = rawVideoExtractor.process_frame_order(video_slice, frame_order=frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error.".format(video_path))
    except Exception as excep:
        print("video path: {} error. Error: {}".format(video_path, excep))
        raise excep
    
    print(torch.from_numpy(video).size())
    return torch.from_numpy(video).squeeze(2)


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# device = 'cpu'
# @st.cache(allow_output_mutation=True, hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
@st.cache_resource
def load_model() -> xllm_models:
    model, vis_processors, _ = load_model_and_preprocess(name="xllm", model_type="pretrain_xllm", is_eval=True, device=device)
    return model, vis_processors

def preprocess(text):
  text = text.replace("\n", "\\n").replace("\t", "\\t")
  return text

def postprocess(text):
  return text.replace("\\n", "\n").replace("\\t", "\t")


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'history' not in st.session_state:
    st.session_state['history'] = []
    st.session_state['history_image'] = []
    st.session_state['input_past'] = "Let's talk about something!"
    st.session_state['input_image'] = None
    st.session_state['audio_time'] = os.path.getmtime(f"{STREAMLIT_RUNNING_PATH}/latest_record.wav")
    st.session_state['baike1'] = {}
    st.session_state['baike2'] = {}

if 'use_image' not in st.session_state:
    st.session_state['use_image'] = False
    st.session_state['use_video'] = False

def get_image():
    # uploaded_file = st.file_uploader("image")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "mp4"])
    if uploaded_file is not None:
        print(uploaded_file.name)
        if "mp4" in uploaded_file.name:
            print('use video: ', uploaded_file.name)
            video_bytes = uploaded_file.read()
            tfile = tempfile.NamedTemporaryFile(prefix='upload_video', suffix='.mp4', delete=False, dir='video/')
            tfile.write(video_bytes)
            
            # cap = cv2.VideoCapture(tfile.name)
            print(uploaded_file.name)
            if len(st.session_state['history_image']) == 0:
                st.session_state['history'].append((video_bytes, '#image#收到。#video'))
                st.session_state['history_image'].append((uploaded_file.name, uploaded_file))
            if uploaded_file.name != st.session_state['history_image'][-1][0]:
                st.session_state['history'].append((video_bytes, '#image#收到。#video'))
                st.session_state['history_image'].append((uploaded_file.name, uploaded_file))
            video = process_video(tfile.name)
            st.session_state['use_video'] = True
            st.session_state['use_image'] = False
            return video, uploaded_file, True
        print('use image: ', uploaded_file.name)
        if len(st.session_state['history_image']) == 0:
            st.session_state['history'].append((uploaded_file, '#image#收到。'))
            st.session_state['history_image'].append((uploaded_file.name, uploaded_file))
        if uploaded_file.name != st.session_state['history_image'][-1][0]:
            st.session_state['history_image'].append((uploaded_file.name, uploaded_file))
            st.session_state['history'].append((uploaded_file, '#image#收到。'))
        st.session_state['use_image'] = True
        st.session_state['use_video'] = False
        return Image.open(uploaded_file).convert("RGB"), uploaded_file, False
    else:
        st.session_state['use_image'] = False
        st.session_state['use_video'] = False
        return None, None, False
    
def on_input_change(text):
    if not (text == st.session_state['input_past']):
        print('sssssssss')
        return None
    st.session_state['input_past']= text
    return text


def get_text():
    raw_image = None
    col1, col2 = st.columns([8,3])
    
        
    with col1:
        
        queryText = st.text_input("Message:", value="", key="input")
        
    with col2:
        state = st.selectbox('Model Prompt',('Pure', "ASR"))
    selected3 = option_menu(None, ["Send", "Image",  "Speech", 'Clear'], 
        icons=['send', 'card-image', "mic", 'gear'], 
        menu_icon="cast", default_index=0, orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "black", "font-size": "15px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#87CEFA"},
        }
    )
    use_video = False
    if selected3 == 'Speech':
        # Record input audio
        st.text("请录入语音")
        # Record Audio Inputs
        
        webrtc_ctx = webrtc_streamer(
            key="sendonly-audio",
            mode=WebRtcMode.SENDONLY,
            in_recorder_factory=recorder_factory,
            media_stream_constraints={
                "audio": True,
                "video": False,
            }
        )


        speeech_choice = st.selectbox('speech',('缓存语音','录入语音', 'test'))
        
        if speeech_choice == '缓存语音':
            speeech_file = f"{STREAMLIT_RUNNING_PATH}/calibrited_latest_record.wav"
        elif speeech_choice == '录入语音':
            if os.path.exists(f"{STREAMLIT_RUNNING_PATH}/latest_record.wav"):
                if os.path.exists(f"{STREAMLIT_RUNNING_PATH}/calibrited_latest_record.wav"):
                    os.remove(f"{STREAMLIT_RUNNING_PATH}/calibrited_latest_record.wav")
                os.system(f"sox {STREAMLIT_RUNNING_PATH}/latest_record.wav -r 16000 "
                        f"-c 1 {STREAMLIT_RUNNING_PATH}/calibrited_latest_record.wav")
            else:
                raise NotImplementedError("Invalid audio input.")
            speeech_file = f"{STREAMLIT_RUNNING_PATH}/calibrited_latest_record.wav"

        else:
            speeech_file = "demo/audio/test_speech.wav"
        st.audio(speeech_file)
        
        SpeechResult = st.button('Input Speech')  

    elif selected3 == 'Image':
        raw_image, uploaded_file_image, use_video = get_image()
        st.session_state['input_image'] = raw_image
        selected3 = 'Send'
        
    elif selected3 == 'Clear':
        # clrResult= st.button('Clear History') 
        # if clrResult:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['history'] = []
        st.session_state['use_image'] = False
        st.session_state['use_video'] = False
        st.session_state['history_image'] = []
        st.session_state['audio_time'] = None
        st.session_state['input_past'] = "Let's talk about something!"
        st.session_state['baike1'] = {}
        st.session_state['input_image'] = None
        
        # st.session_state['baike2'] = {}
        st.write('Clear successfully!')

    if queryText != st.session_state['input_past']:  
        st.session_state['input_past'] =  queryText 
        # if selected3 == 'Send' and btnResult:                 
        if selected3 == 'Send':
            return queryText, None, state, st.session_state['input_image'], st.session_state['use_video'] 
    # else:
    #     if st.session_state['input_past'] != "Let's talk about something!":
    #         st.write('你需要换一个输入。')
    if selected3 == 'Speech' and SpeechResult:
        return queryText, speeech_file, state, st.session_state['input_image'], st.session_state['use_video'] 

    return False, None, state, None, False




# @st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})

def main(model, vis_processors, device):

    container = st.container()
    st.write('')
    st.write('')
    user_input, speech, model_state, raw_image, use_video = get_text()
    
    
    if raw_image is None:
        image = None
    elif not use_video:
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    else:
        image = raw_image
    with container:
        
        if len(st.session_state['history']) > 0:
            for i, (query1, response1) in enumerate(st.session_state['history']):
                if '#image#收到。' in response1:
                    if '#video' in response1:
                        message('上传一段视频', avatar_style="fun-emoji", is_user=True, key=str(i) + "_user")
                    else:
                        message('上传一张图片', avatar_style="fun-emoji", is_user=True, key=str(i) + "_user")
                    # st.write('<style>img {max-width: 50%; height: auto;}</style>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([5,7,1])
                    with col2:
                        if '#video' in response1:
                            st.video(query1)
                        else:
                            st.image(query1, channels="BGR", use_column_width=True)
                        # st.image(img, use_column_width=True)
                        st.markdown(
                            """
                            <style>
                            img {

                                max-width: 100%;
                            
                            }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )
                    message('收到。', avatar_style="bottts", key=str(i))
                else:
                    message(query1, avatar_style="fun-emoji", is_user=True, key=str(i) + "_user")
                    message(response1, avatar_style="bottts", key=str(i))
        if user_input or speech:
            st.session_state['past'].append(user_input)
            if st.session_state['use_image']:
                if model_state == 'Pure':
                        query = user_input
                else:
                    query = user_input
            else:
                if user_input == '你好':
                    query = user_input
                else:
                    if model_state == 'Pure':
                        query = user_input
                    else:
                        query =  user_input
            history = st.session_state['history']
            if speech is None:
                message(user_input, avatar_style="fun-emoji", is_user=True, key=str(len(history)) + "_user")
                st.write("AI正在回复:")
                    

            if speech is None:
                print('use image', st.session_state['use_image'])
                if not history:
                    prompt = query
                else:
                    prompt = ""
                    for i, (old_query, response) in enumerate(history):
                        if '#image#收到。' in response:
                            continue
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                print("prompt: ", prompt)
                print('use video', st.session_state['use_video'])
                
                with st.empty():
                    for response in model.generate_demo_stream({"image": image, "prompt": prompt, "use_image": st.session_state['use_image'], "use_video": st.session_state['use_video']}):
                        st.write(response)
                
                st.session_state['history'].append((user_input, response))

            else:
                if model_state == 'ASR':
                    message('请识别输入语音', avatar_style="fun-emoji", is_user=True, key=str(len(history)) + "_user")
                   
                    prompt = '请忠实地识别该语音'
                    # prompt = '请识别这段语音并翻译成英文'
                    speech_feature = get_ASRFeature(speech)
                    st.write("AI正在回复:")
                    with st.empty():
                        for response in model.generate_demo_stream({"image": image, "prompt": prompt, "speech": speech_feature, "use_image": False, "use_video": False}):
                            st.write(response)

              
                    st.session_state['history'].append(('请识别输入语音', response))

                    print(response)
            
            st.experimental_rerun()
            

    with container.empty():
            
        if not st.session_state['past']:
            st.session_state['past'].append("让我们开始聊天吧！")
            message(st.session_state['past'][0], key=str(-1))
            st.session_state['past'] = []
 

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


if __name__ == '__main__':
    model, vis_processors = load_model()
    # vis_processors = load_model_1()
    st.header("UniDial Chat Bot: 百灵（Lark）")
    main(model, vis_processors, device)
    

    