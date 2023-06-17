# Training

X-LLM connects multiple pre-trained single-modal encoders (such as ViT-g visual encoder) and large language model ChatGLM, using X2L interfaces. We consider a three-stage training procedure:
- Stage 1: Converting Multimodal Information. Convert multimodal information into foreign languages through X2L interfaces, only X2L interfaces are updated
- Stage 2: Aligning X2L Representations with the LLM. Inject foreign languages into LLM, only X2L interfaces are updated.
- Stage 3: Integrating Multiple Modalities. Integrating multi-modalities, only the adapters in X2L interfaces are updated.

## Before training
1. You should replace the related absolute paths of datasets. Please see [README_DATA.md](https://github.com/phellonchen/X-LLM/blob/main/README_DATA.md) for details.
2. Download the model checkpoint of [ChatGLM](https://github.com/THUDM/ChatGLM-6B), and replace the opt_model path in xllm/projects/train/\*.yaml

## Stage 1
We find that the Q-former module trained on English image-text data can be transferred to other languages. So we use the Q-Former parameters trained in the first stage of BLIP2 to initialize the Q-Former. And we start the second stage of training the model directly.

We will release the stage 1's script to train the model from sractch as soon as possible.

## Stage 2

### Train Image Interface
```
bash run_scripts/train/image_interface_stage2.sh
```

### Train Video Interface
We use the Q-Former of Image Interface to initialize the Q-Former of Video Interface.

Firstly, you should use the checkpoint saved in training Image Interface in stage 2 as the pretained model. (replace the value path of the key "pretrained" in xllm/projects/video_interface_stage2.yaml).

Secondly, 
```
bash run_scripts/train/video_interface_stage2.sh
```

### Train Speech Interface
For efficient training, we pre extract speech features for model training, rather than using the speech encoder to extract features during the training process. 

```
bash run_scripts/train/speech_interface_stage2.sh
```

The speech encoder we used is not included in this project. Please see this [project]() to review the detailed structure and training method of the speech encoder.

## Stage 3

### Multimodal Instruction Training
```
bash run_scripts/train/instruction_stage3.sh
```

# Evaluation








