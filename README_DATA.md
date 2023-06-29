# Datasets

We will update this file as soon as possible！

We provide links to download our preprocessed dataset. If you would like to process the data on your own, we will soon provide scripts for you to do so. 

Note that you should replace the image/video/speech file paths in the json files according to your storage path.

And please use your own file path to replace the original path in xllm/configs/datasets/\*/\*.yaml or xllm/projects/train/\*.yaml 

## Image Interface
The pretraining datasets used in X-LLM are all publicly available. Here we provide the public links to these data, it is recommended that you download images pf the data from the links first, and then link the image paths with the downloaded dataset json (Chinese) we provided.

<table border="1" width="100%">
    <tr align="center">
        <th>Dataset</th><th>Image</th><th>Data</th><td>Language</td>
    </tr>
    <tr align="center">
        <td>CC3M</td><td><a href="https://github.com/google-research-datasets/conceptual-captions">Image Url</a></td><td><a href="https://drive.google.com/file/d/1Mb--df0gCDTYt0kpnP24CPMAZzYrPlrS/view?usp=drive_link">Data Json</a></td><td>ZH</td>
    </tr>
    <tr align="center">
        <td>MSCOCO</td><td><a href="https://cocodataset.org/">Image Url</a></td><td><a href="https://drive.google.com/file/d/1PdkguGtBXosvzTgwGXD5uYZisQ_zLAdH/view?usp=drive_link">Data Json</a></td><td>ZH</td>
    </tr>
    <tr align="center">
        <td>Visual Genome</td><td><a href="https://visualgenome.org/">Image Url</a></td><td><a href="https://drive.google.com/file/d/1ofZE2fIaYxBSZy36-9V-zxrt9JUQM1X2/view?usp=drive_link">Data Json</a></td><td>ZH</td>
    </tr>
    <tr align="center">
        <td>Flickr30k</td><td><a href="http://shannon.cs.illinois.edu/DenotationGraph/">Image Url</a></td><td><a href="https://drive.google.com/file/d/1ofZE2fIaYxBSZy36-9V-zxrt9JUQM1X2/view?usp=drive_link">Data Json</a></td><td>ZH</td>
    </tr>
    <tr align="center">
        <td>SBU</td><td><a href="https://www.cs.rice.edu/~vo9/sbucaptions/">Image Url</a></td><td><a href="https://drive.google.com/file/d/1-dvv9Pqv_uY1DjaTKlELM9KyYUq68yek/view?usp=drive_link">Data Json</a></td><td>ZH</td>
    </tr>
    <tr align="center">
        <td>AI Challenger captions</td><td><a href="https://github.com/AIChallenger/AI_Challenger_2017">Image Url</a></td><td><a href="https://drive.google.com/file/d/1qxufClVlcz4S4acweqgkpcFbEyax7Rx7/view?usp=drive_link">Data Json</a></td><td>ZH</td>
    </tr>
    <tr align="center">
        <td>Wukong captions</td><td><a href="https://wukong-dataset.github.io/wukong-dataset/">Image Url</a></td><td><a href="https://drive.google.com/file/d/14-GWHbhVWSIDWdeIzZLRU15oURqLR0kC/view?usp=drive_link">Data Json</a></td><td>ZH</td>
    </tr>
</table>

Please note that for the Wukong dataset, we filtered the first 50 million images using CLIP and only kept samples with a visual-textual similarity score greater than 0.475. Additionally, you will need to pair the captions with the corresponding images based on the image captions.
<br></br>

Data Format
```
[
    {
        "image": "train2014/COCO_train2014_000000013356.jpg",
        "caption": [
            "一个站在玻璃附近的白衣男子",
            "一个人在破旧的浴室里穿着防护服和面具",
            "一个人从头到脚穿着白色涂在房间里",
            "浴室正在装修，一个人在墙上画画",
            "一个穿着防护服的人在房间里工作"
        ],
        "image_id": "train2014/COCO_train2014_000000013356.jpg",
        "dataset": "coco_zh"
    },
]
```
or
```
[
    {
        "image": "/raid/cfl/en_pretraining/data/images/sbu/pythonDownload/subpic/5eda85e140.jpg",
        "caption": "谢菲尔德公园花园苏塞克斯湖边的老树",
        "image_id": "5eda85e140.jpg",
        "dataset": "sbu_zh"
    },
]
```
We do not use the item "image_id", which is the same as "image" most cases.
Note that you should replace the image paths in the json files according to your storage path.

## Speech Interface
We provide the public links to speech data (*.wav & feats), it is recommended that you download the data from the links first, and then link the speech data paths with the downloaded dataset json we provided.

<table border="1" width="100%">
    <tr align="center">
        <th>Dataset</th><th>Audio/Features</th><th>Data</th><th>Language</th>
    </tr>
    <tr align="center">
        <td>AISHELL-2</td><td><a href="">Audio/Features</a></td><td><a href="">Data Json</a></td><td>ZH</td>
    </tr>
    <tr align="center">
        <td>VSDial-CN</td><td><a href="">Audio/Features</a></td><td><a href="">Data Json</a></td><td>ZH</td>
    </tr>
</table>
<br></br>

## Video Interface
The pretraining datasets used in X-LLM are all publicly available. Here we provide the public links to these data, it is recommended that you download video pf the data from the links first, and then link the video paths with the downloaded dataset json (Chinese) we provided.


<table border="1" width="100%">
    <tr align="center">
        <th>Dataset</th><th>Video</th><th>Data</th>
    </tr>
    <tr align="center">
        <td>MSRVTT</td><td><a href="https://github.com/ArrowLuo/CLIP4Clip">Video Url</a></td><td><a href="">Data Json</a></td>
    </tr>
    <tr align="center">
        <td>ActivityNet</td><td><a href="http://activity-net.org/download.html">Video Url</a></td><td><a href="">Data Json</a></td>
    </tr>
</table>
<br></br>

## Evaluation 
We provide the Chinese version of LLaVA test, which is an evaluation dataset with 30 unseen images is constructed: each image is assocaited with three types of instructions: conversation, detailed description and complex reasoning.
