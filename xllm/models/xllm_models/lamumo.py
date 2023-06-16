"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import re
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import pdb

from xllm.common.registry import registry
from xllm.models.xllm_models.blip2 import Blip2Base, disabled_train
from xllm.models.xllm_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer, AutoModel
from xllm.models.xllm_models.modeling_chatglm import ChatGLMForConditionalGeneration
from xllm.models.xllm_models.tokenization_chatglm import ChatGLMTokenizer
import base64
import numpy as np
from transformers import BertModel
from transformers import (
    BertModel,
    BertTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertForNextSentencePrediction
)

@registry.register_model("lamumo")
class Lamumo(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from xllm.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_lamumo": "configs/models/blip2/lamumo.yaml",
    }

    def __init__(
        self,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="THUDM/chatglm-6b",
        prompt="",
        max_txt_len=32,
        forward_mode="",
        split_tasks="",
        asr_loss_scale=1.0,
        vd_loss_scale=1.0,
        vsd_loss_scale=1.0,
        speech_adaptor_type="bert",
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision, vision_hidden = self.init_vision_encoder(
            img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False               
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.opt_tokenizer = ChatGLMTokenizer.from_pretrained(opt_model)
        self.opt_model = ChatGLMForConditionalGeneration.from_pretrained(
            opt_model, torch_dtype=torch.bfloat16
        ).half()
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.opt_model.gradient_checkpointing_enable()
        # self.opt_model.enable_input_require_grads()
        self.ignore_pad_token_for_loss = True

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )
        self.video_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.video_query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.video_query_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)

        self.max_txt_len = max_txt_len
    
        # Some settings about the training of the third stage, recently added by Minglun Han
        self.forward_mode = forward_mode
        self.split_tasks = [x.strip() for x in split_tasks.split(",")]
        self.asr_loss_scale = asr_loss_scale
        self.vd_loss_scale = vd_loss_scale
        self.vsd_loss_scale = vsd_loss_scale
        self.speech_adaptor_type = speech_adaptor_type
        if self.speech_adaptor_type == "fc":
            self.speech_feat_proj = nn.Linear(
                512, self.opt_model.config.hidden_size,
            )   # default dimension of CIF-based speech feats is 1024.
        elif self.speech_adaptor_type == "bert":
            self.speech_bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            self.speech_bert_adaptor = BertModel.from_pretrained("bert-base-chinese")
            self.bert_hidden_size = self.speech_bert_adaptor.config.hidden_size
            self.pre_bert_proj = nn.Linear(512, self.bert_hidden_size)
            self.post_bert_proj = nn.Linear(
                self.bert_hidden_size, self.opt_model.config.hidden_size
            )
        else:
            raise ValueError("Unsupported adaptor type.")

        if self.forward_mode == "split":
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
                param.data = param.data.bfloat16()
            for name, param in self.speech_bert_adaptor.named_parameters():
                param.requires_grad = False
                param.data = param.data.bfloat16() 
            for name, param in self.pre_bert_proj.named_parameters():
                param.requires_grad = False
                param.data = param.data.bfloat16() 
            # for name, param in self.post_bert_proj.named_parameters():
            #     param.requires_grad = False
            #     param.data = param.data.bfloat16() 
            # for name, param in self.opt_proj.named_parameters():
            #     param.requires_grad = False
            #     param.data = param.data.bfloat16() 
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
                param.data = param.data.bfloat16() 



    def prompt_wrap(self, img_embeds, atts_img, prompt, use_speech=True, video=False):
        if use_speech:
            special_token = '<SpeechHere>'
        else:
            special_token = '<ImageHere>'
        if video:
            special_token = '<VideoHere>'
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split(special_token)
            p_before_tokens = self.opt_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.opt_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.opt_model.transformer.word_embeddings(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.opt_model.transformer.word_embeddings(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img
        
    def vl_forward(self, samples):
        # for automatic speech recognition
        if samples is None:
            return torch.tensor(0.0)

        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        image_embeds = self.opt_proj(query_output.last_hidden_state)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.bool).to(image.device)
    
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):

            image_prompt = '<Image><ImageHere></Image>'
            image_embeds, image_atts = self.prompt_wrap(image_embeds, image_atts, image_prompt, use_speech=False)

            opt_tokens = self.preprocess_function_train(samples, image_atts.device, src_txt_len=32, max_tgt_len=200)

            # pdb.set_trace()

            empty_targets = (
                torch.ones(image_atts.size(),
                           dtype=torch.long).to(image_atts.device).fill_(0)
            )

            # pdb.set_trace()

            opt_tokens['input_ids'] = torch.cat(
                [empty_targets, opt_tokens['input_ids']], dim=1
            ).to(image_embeds.device)
            opt_tokens['labels'] = torch.cat(
                [empty_targets.fill_(-100), opt_tokens['labels']], dim=1
            ).to(image_embeds.device)
  
            outputs = self.opt_model(
                **opt_tokens,
                input_image=image_embeds,
                return_dict=True,
            )
            loss = outputs.loss

        return loss
    
    def video_forward(self, samples):
        # for automatic speech recognition
        if samples is None:
            return torch.tensor(0.0)

        image = samples["video"]
        batch_size, max_frame, image_channel, resolution, _ = image.size()
        image = image.view(batch_size * max_frame, image_channel, resolution, resolution)
        image_embeds = self.ln_vision(self.visual_encoder(image))
        

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.video_query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        image_embeds = self.video_proj(query_output.last_hidden_state)
        _, per_image_length, image_hidden_size = image_embeds.size()
        image_embeds = image_embeds.view(batch_size, max_frame, per_image_length, image_hidden_size).view(batch_size, max_frame*per_image_length, image_hidden_size)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.bool).to(image.device)
    
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):

            image_prompt = '<Video><VideoHere></Video>'
            image_embeds, image_atts = self.prompt_wrap(image_embeds, image_atts, image_prompt, use_speech=False, video=True)

            opt_tokens = self.preprocess_function_train(samples, image_atts.device, src_txt_len=32, max_tgt_len=32)

          

            empty_targets = (
                torch.ones(image_atts.size(),
                           dtype=torch.long).to(image_atts.device).fill_(0)
            )

            
            opt_tokens['input_ids'] = torch.cat(
                [empty_targets, opt_tokens['input_ids']], dim=1
            ).to(image_embeds.device)
            opt_tokens['labels'] = torch.cat(
                [empty_targets.fill_(-100), opt_tokens['labels']], dim=1
            ).to(image_embeds.device)
  
            outputs = self.opt_model(
                **opt_tokens,
                input_image=image_embeds,
                return_dict=True,
            )
            loss = outputs.loss

        return loss

    def asr_forward(self, samples):
        # for automatic speech recognition
        if samples is None:
            return torch.tensor(0.0)

        speech_embeds = samples["speech_input"]  # B x T x C

        speech_embeds_attn_mask = (speech_embeds.sum(-1) != 0.0).long()  # B x T
    
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if self.speech_adaptor_type == "fc":
                speech_embeds = self.speech_feat_proj(speech_embeds)  # B x T x C
                speech_embeds = nn.ReLU()(speech_embeds)
            else:
                speech_embeds = nn.ReLU()(self.pre_bert_proj(speech_embeds))  # B x T x C
                sbert_outputs = self.speech_bert_adaptor(
                    inputs_embeds=speech_embeds,
                    attention_mask=speech_embeds_attn_mask,
                )
                speech_embeds = nn.ReLU()(self.post_bert_proj(sbert_outputs[0]))  # B x T x C


            speech_prompt = '<Speech><SpeechHere></Speech>'
            speech_embeds, speech_embeds_attn_mask = self.prompt_wrap(speech_embeds, speech_embeds_attn_mask, speech_prompt)

            opt_tokens = self.preprocess_function_train(samples, speech_embeds_attn_mask.device, src_txt_len=32, max_tgt_len=32)

          

            empty_targets = (
                torch.ones(speech_embeds_attn_mask.size(),
                           dtype=torch.long).to(speech_embeds.device).fill_(0)
            )

          
            opt_tokens['input_ids'] = torch.cat(
                [empty_targets, opt_tokens['input_ids']], dim=1
            ).to(speech_embeds.device)
            opt_tokens['labels'] = torch.cat(
                [empty_targets.fill_(-100), opt_tokens['labels']], dim=1
            ).to(speech_embeds.device)
  
            outputs = self.opt_model(
                **opt_tokens,
                input_speech=speech_embeds,
                return_dict=True,
            )
            loss = outputs.loss

        return loss 
    
    def masr_forward(self, samples):
        # for automatic speech recognition
        if samples is None:
            return torch.tensor(0.0)
        
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        image_embeds = self.opt_proj(query_output.last_hidden_state)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.bool).to(image.device)

        speech_embeds = samples["speech_input"]  # B x T x C

        speech_embeds_attn_mask = (speech_embeds.sum(-1) != 0.0).long()  # B x T
    
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if self.speech_adaptor_type == "fc":
                speech_embeds = self.speech_feat_proj(speech_embeds)  # B x T x C
                speech_embeds = nn.ReLU()(speech_embeds)
            else:
                speech_embeds = nn.ReLU()(self.pre_bert_proj(speech_embeds))  # B x T x C
                sbert_outputs = self.speech_bert_adaptor(
                    inputs_embeds=speech_embeds,
                    attention_mask=speech_embeds_attn_mask,
                )
                speech_embeds = nn.ReLU()(self.post_bert_proj(sbert_outputs[0]))  # B x T x C

            image_prompt = '<Image><ImageHere></Image>'
            image_embeds, image_atts = self.prompt_wrap(image_embeds, image_atts, image_prompt, use_speech=False)

            speech_prompt = '<Speech><SpeechHere></Speech>'
            speech_embeds, speech_embeds_attn_mask = self.prompt_wrap(speech_embeds, speech_embeds_attn_mask, speech_prompt)

            opt_tokens = self.preprocess_function_train(samples, speech_embeds_attn_mask.device, src_txt_len=32, max_tgt_len=32)

            image_empty_targets = (
                torch.ones(image_atts.size(),
                           dtype=torch.long).to(speech_embeds.device).fill_(0)
            )

            speech_empty_targets = (
                torch.ones(speech_embeds_attn_mask.size(),
                           dtype=torch.long).to(speech_embeds.device).fill_(0)
            )

            opt_tokens['input_ids'] = torch.cat(
                [image_empty_targets, speech_empty_targets, opt_tokens['input_ids']], dim=1
            ).to(speech_embeds.device)
            opt_tokens['labels'] = torch.cat(
                [image_empty_targets.fill_(-100), speech_empty_targets.fill_(-100), opt_tokens['labels']], dim=1
            ).to(speech_embeds.device)
  
            outputs = self.opt_model(
                **opt_tokens,
                input_speech=speech_embeds,
                input_image=image_embeds,
                return_dict=True,
            )
            loss = outputs.loss

        return loss 
    
    def vsl_forward(self, samples):
  
        if samples is None:
            return torch.tensor(0.0)
        
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        image_embeds = self.opt_proj(query_output.last_hidden_state)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.bool).to(image.device)

        speech_embeds = samples["speech_input"]  # B x T x C

        speech_embeds_attn_mask = (speech_embeds.sum(-1) != 0.0).long()  # B x T
    
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if self.speech_adaptor_type == "fc":
                speech_embeds = self.speech_feat_proj(speech_embeds)  # B x T x C
                speech_embeds = nn.ReLU()(speech_embeds)
            else:
                speech_embeds = nn.ReLU()(self.pre_bert_proj(speech_embeds))  # B x T x C
                sbert_outputs = self.speech_bert_adaptor(
                    inputs_embeds=speech_embeds,
                    attention_mask=speech_embeds_attn_mask,
                )
                speech_embeds = nn.ReLU()(self.post_bert_proj(sbert_outputs[0]))  # B x T x C

            image_prompt = '<Image><ImageHere></Image>'
            image_embeds, image_atts = self.prompt_wrap(image_embeds, image_atts, image_prompt, use_speech=False)

            speech_prompt = '<Speech><SpeechHere></Speech>'
            speech_embeds, speech_embeds_attn_mask = self.prompt_wrap(speech_embeds, speech_embeds_attn_mask, speech_prompt)

            opt_tokens = self.preprocess_function_train(samples, speech_embeds_attn_mask.device, src_txt_len=256, max_tgt_len=32)

            image_empty_targets = (
                torch.ones(image_atts.size(),
                           dtype=torch.long).to(speech_embeds.device).fill_(0)
            )

            speech_empty_targets = (
                torch.ones(speech_embeds_attn_mask.size(),
                           dtype=torch.long).to(speech_embeds.device).fill_(0)
            )

            opt_tokens['input_ids'] = torch.cat(
                [image_empty_targets, speech_empty_targets, opt_tokens['input_ids']], dim=1
            ).to(speech_embeds.device)
            opt_tokens['labels'] = torch.cat(
                [image_empty_targets.fill_(-100), speech_empty_targets.fill_(-100), opt_tokens['labels']], dim=1
            ).to(speech_embeds.device)
  
            outputs = self.opt_model(
                **opt_tokens,
                input_speech=speech_embeds,
                input_image=image_embeds,
                return_dict=True,
            )
            loss = outputs.loss

        return loss 

    def regather_samples(self, samples, task_id):
        binary_index = (samples["task_id"] == task_id).int()
        index = torch.nonzero(binary_index)

        if index.size(0) == 0:
            return None
        else:
            index = torch.squeeze(index, dim=-1)    # B

        output_samples = dict()
        for key, value in samples.items():
            if torch.is_tensor(value):
                new_value = torch.index_select(value, dim=0, index=index)
                output_samples[key] = new_value
            elif isinstance(value, list):
                index_list = index.tolist()
                new_value = []
                for ind, v in enumerate(value):
                    if ind in index_list:
                        new_value.append(v)
                output_samples[key] = new_value
            else:
                continue

        return output_samples      

    def forward(self, samples):
        vl_loss = torch.tensor(0.0)
        asr_loss = torch.tensor(0.0)
        masr_loss = torch.tensor(0.0)
        vsd_loss = torch.tensor(0.0)
        video_loss = torch.tensor(0.0)

        # asr,masr,vl,vsd
        if self.forward_mode == "split":
            if "vl" in self.split_tasks:
                vl_samples = self.regather_samples(samples, task_id=1)
                vl_loss = self.vl_forward(vl_samples)

            if "asr" in self.split_tasks:
                asr_samples = self.regather_samples(samples, task_id=2)
                asr_loss = self.asr_forward(asr_samples)

            if "masr" in self.split_tasks:
                masr_samples = self.regather_samples(samples, task_id=3)
                masr_loss = self.masr_forward(masr_samples)

            if "vsd" in self.split_tasks:
                vsd_samples = self.regather_samples(samples, task_id=4)
                vsd_loss = self.vsl_forward(vsd_samples)

            if "video" in self.split_tasks:
                video_samples = self.regather_samples(samples, task_id=5)
                video_loss = self.video_forward(video_samples)

            loss = vl_loss + asr_loss + masr_loss + vsd_loss + video_loss
      
        return {"loss": loss}

    

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        use_image = True
        image = samples["image"]
        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):          
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = samples["text_input"]
                # prompt = self.prompt

            if isinstance(prompt, str):
                prompt = [prompt] * image.size(0)
            else:
                assert len(prompt) == image.size(
                    0
                ), "The number of prompts must be equal to the batch size."

            opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt").to(image.device)
            input_ids = opt_tokens.input_ids
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):

                outputs = self.opt_model.generate(
                    input_ids=input_ids,
                    query_embeds=query_embeds,
                    attention_mask=attention_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )

                prompt_length = opt_tokens.input_ids.shape[1]
                output_text = self.opt_tokenizer.batch_decode(
                    outputs[:, prompt_length:], skip_special_tokens=True
                )
                output_text = [text.strip() for text in output_text]
                return output_text
            

    @torch.no_grad()
    def asr_generate(
            self,
            samples,
            num_beams=5,
            max_length=30,
            min_length=1,
            **kwargs,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        max_length = 80
        num_beams = 1
        do_sample = False # True
        top_p = 1.0 # 0.7
        temperature = 0.95 # 0.95
        repetition_penalty = 1.0

        # max_length=80
        # num_beams=1
        # do_sample=True # True
        # top_p=0.7
        # temperature=0.95 # 0.95
     
        # gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
        #               "temperature": temperature, **kwargs}
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            **kwargs
        }

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        # get speech inputs
        speech_embeds = samples["speech_input"]  # B x T x C
        speech_embeds_attn_mask = (speech_embeds.sum(-1) != 0.0).long()  # B x T
    

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            
            if self.speech_adaptor_type == "fc":
                speech_embeds = self.speech_feat_proj(speech_embeds)  # B x T x C
                speech_embeds = nn.ReLU()(speech_embeds)
            else:
                speech_embeds = nn.ReLU()(self.pre_bert_proj(speech_embeds))  # B x T x C
                sbert_outputs = self.speech_bert_adaptor(
                    inputs_embeds=speech_embeds,
                    attention_mask=speech_embeds_attn_mask,
                )
                speech_embeds = nn.ReLU()(self.post_bert_proj(sbert_outputs[0]))  # B x T x C

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = samples["text_input"]

            speech_prompt = '<Speech><SpeechHere></Speech>'
            speech_embeds, speech_embeds_attn_mask = self.prompt_wrap(speech_embeds, speech_embeds_attn_mask, speech_prompt)

            opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt", padding=True).to(speech_embeds.device)

            opt_tokens = opt_tokens.to(speech_embeds.device)
            context_length = opt_tokens.input_ids.size(1)

            empty_targets = (
                torch.ones(speech_embeds_attn_mask.size(), dtype=torch.long).to(speech_embeds.device).fill_(0)
            )

            opt_tokens['input_ids'] = torch.cat([empty_targets, opt_tokens.input_ids], dim=1)
            # opt_tokens['attention_mask'] = torch.cat([speech_embeds_attn_mask, opt_tokens.attention_mask], dim=1)
            opt_tokens = opt_tokens.to(speech_embeds.device)
            context_length = opt_tokens.input_ids.size(1)
            del opt_tokens['attention_mask']
            del opt_tokens['position_ids']

    

            outputs = self.opt_model.generate(
                **opt_tokens, **gen_kwargs,
                input_speech=speech_embeds
            )
            # outputs_list = outputs.tolist()
            # response = [self.opt_tokenizer.decode(_[context_length-2:]).strip().replace("[[训练时间]]", "2023年") for _ in outputs_list]
            try:
                # outputs = outputs.tolist()[0][context_length - 4:]
                outputs = outputs.tolist()[0][context_length - 2:]
                response = self.opt_tokenizer.decode(outputs)
                response = response.strip()
                response = response.replace("[[训练时间]]", "2023年")
                response = [response]
            except:
                response = ['Error']

        return response


    @torch.no_grad()
    def generate_demo(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        **kwargs
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        max_length=2048
        num_beams=1
        do_sample=True
        top_p=0.7
        temperature=0.95
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, **kwargs}

        use_image = samples["use_image"]
        image = samples["image"]
        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):          
            if use_image:
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                    image.device
                )

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                inputs_opt = self.opt_proj(query_output.last_hidden_state)
                atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.bool).to(image.device)

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = samples["text_input"]
                # prompt = self.prompt

            # if isinstance(prompt, str):
            #     prompt = [prompt] * image.size(0)
            # else:
            #     assert len(prompt) == image.size(
            #         0
            #     ), "The number of prompts must be equal to the batch size."

            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            opt_tokens = self.opt_tokenizer([prompt], return_tensors="pt", padding=True).to(image.device)
            opt_tokens = opt_tokens.to(image.device)
            context_length = opt_tokens.input_ids.size(1)    
            
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                if use_image:
                    empty_targets = (
                        torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(0)
                    )
                    opt_tokens['input_ids'] = torch.cat([empty_targets, opt_tokens.input_ids], dim=1)
                    # opt_tokens['attention_mask'] = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
                    opt_tokens = opt_tokens.to(image.device)
                    context_length = opt_tokens.input_ids.size(1)
                    print('context_length: ', context_length)
                    outputs = self.opt_model.generate(
                            **opt_tokens, **gen_kwargs, inputs_opt=inputs_opt,
                        )
                else:
                    outputs = self.opt_model.generate(
                        **opt_tokens, **gen_kwargs, 
                    )
                print('output length: ', len(outputs.tolist()[0]))
                # print(outputs.tolist()[0])
                # response = self.opt_tokenizer.decode(outputs.tolist()[0])
                # print(response)
                outputs = outputs.tolist()[0][context_length -  2:]
                response = self.opt_tokenizer.decode(outputs)
                response = response.strip()
                response = response.replace("[[训练时间]]", "2023年")
                return [response]
            
    @torch.no_grad()
    def generate_demo_stream(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        **kwargs
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        # max_length=2048
        # num_beams=1
        # do_sample=True
        # top_p=0.7
        # temperature=0.95
        # gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
        #               "temperature": temperature, **kwargs}


        max_length=2048
        # num_beams=1
        # do_sample=True
        # top_p=0.9
        # temperature=0.95
        gen_kwargs = {"max_length": max_length, "penalty_alpha": 0.6, "top_k": 4, **kwargs}

        use_image = samples["use_image"]
        image = samples["image"]

        use_video = samples["use_video"]
        use_speech = False
        if 'speech' in samples.keys():
            use_speech = True
        if use_video:
            image = torch.tensor(image).half().to(self.device)
            frame = image.size(1)
            image_resolution = image.size(-1)
            image = image.view(frame, 3, image_resolution, image_resolution)
      
        # print('stream generate image size: ', image.size())
        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):          
            if use_speech:
                if '请忠实地识别该语音' in samples["prompt"]:

                    max_length = 2000
                    num_beams = 1
                    do_sample = False # True
                    top_p = 1.0 # 0.7
                    temperature = 0.95 # 0.95
                    repetition_penalty = 1.0
                    gen_kwargs = {
                        "max_length": max_length,
                        "num_beams": num_beams,
                        "do_sample": do_sample,
                        "top_p": top_p,
                        "temperature": temperature,
                        "repetition_penalty": repetition_penalty,
                        **kwargs
                    }
                
                waveform = base64.b64decode(samples["speech"] )
                waveform = np.frombuffer(waveform, dtype=np.float32)
                speech_embeds = torch.tensor(waveform, dtype=torch.float32).unsqueeze(dim=0).to(self.device)
                speech_embeds = speech_embeds.view(1, -1, 512)
                 # B x T x C
                speech_embeds_attn_mask = (speech_embeds.sum(-1) != 0.0).long()  # B x T

                if use_image or use_video:
                    image_embeds = self.ln_vision(self.visual_encoder(image))
                    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                        image.device
                    )
                    if use_image:
                        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                        query_output = self.Qformer.bert(
                            query_embeds=query_tokens,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            return_dict=True,
                        )
                        image_embeds = self.opt_proj(query_output.last_hidden_state)
                    if use_video:
                        query_tokens = self.video_query_tokens.expand(image_embeds.shape[0], -1, -1)
                        query_output = self.Qformer.bert(
                            query_embeds=query_tokens,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            return_dict=True,
                        )
                        image_embeds = self.video_proj(query_output.last_hidden_state)
                
                        _, image_embeds_length, image_embeds_hidden = image_embeds.size()
                        image_embeds = image_embeds.view(1, frame, image_embeds_length, image_embeds_hidden)
                        image_embeds = image_embeds.view(1, frame*image_embeds_length, image_embeds_hidden)

                    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.bool).to(image.device)


                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            
                    if self.speech_adaptor_type == "fc":
                        speech_embeds = self.speech_feat_proj(speech_embeds)  # B x T x C
                        speech_embeds = nn.ReLU()(speech_embeds)
                    else:
                        speech_embeds = nn.ReLU()(self.pre_bert_proj(speech_embeds))  # B x T x C
                        sbert_outputs = self.speech_bert_adaptor(
                            inputs_embeds=speech_embeds,
                            attention_mask=speech_embeds_attn_mask,
                        )
                        speech_embeds = nn.ReLU()(self.post_bert_proj(sbert_outputs[0]))  # B x T x C

                    if "prompt" in samples.keys():
                        prompt = samples["prompt"]
                    else:
                        prompt = samples["text_input"]

                    speech_prompt = '<Speech><SpeechHere></Speech>'
                    speech_embeds, speech_embeds_attn_mask = self.prompt_wrap(speech_embeds, speech_embeds_attn_mask, speech_prompt)

                    if use_image or use_video:
                        image_prompt = '<Image><ImageHere></Image>'
                        if use_video:
                            image_prompt = '<Video><VideoHere></Video>'
                        image_embeds, image_atts = self.prompt_wrap(image_embeds, image_atts, image_prompt, use_speech=False, video=use_video)
                        empty_targets_image = (
                            torch.ones(image_atts.size(), dtype=torch.long).to(speech_embeds.device).fill_(0)
                        )

                    opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt", padding=True).to(speech_embeds.device)

                    opt_tokens = opt_tokens.to(speech_embeds.device)
                    context_length = opt_tokens.input_ids.size(1)

                    empty_targets = (
                        torch.ones(speech_embeds_attn_mask.size(), dtype=torch.long).to(speech_embeds.device).fill_(0)
                    )
                    
                    opt_tokens['input_ids'] = torch.cat([empty_targets, opt_tokens.input_ids], dim=1)
                    if use_image or use_video:
                        opt_tokens['input_ids'] = torch.cat([empty_targets_image, opt_tokens.input_ids], dim=1)
                    # opt_tokens['attention_mask'] = torch.cat([speech_embeds_attn_mask, opt_tokens.attention_mask], dim=1)
                    opt_tokens = opt_tokens.to(speech_embeds.device)
                    context_length = opt_tokens.input_ids.size(1)
                    del opt_tokens['attention_mask']
                    del opt_tokens['position_ids']

                    if use_image or use_video:
                        for outputs in self.opt_model.stream_generate(**opt_tokens, **gen_kwargs, input_image=image_embeds,input_speech=speech_embeds,):
                            outputs = outputs.tolist()[0][context_length -  2:]
                            response = self.opt_tokenizer.decode(outputs)
                            response = self.process_response(response)
                        
                            yield response
                    else:
                        for outputs in self.opt_model.stream_generate(**opt_tokens, **gen_kwargs, input_speech=speech_embeds,):
                            outputs = outputs.tolist()[0][context_length -  2:]
                            response = self.opt_tokenizer.decode(outputs)
                            response = self.process_response(response)
                        
                            yield response

            else:    
                if use_image or use_video:
                    image_embeds = self.ln_vision(self.visual_encoder(image))
                    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                        image.device
                    )
                    if use_image:
                        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                        query_output = self.Qformer.bert(
                            query_embeds=query_tokens,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            return_dict=True,
                        )
                        image_embeds = self.opt_proj(query_output.last_hidden_state)
                    if use_video:
                        query_tokens = self.video_query_tokens.expand(image_embeds.shape[0], -1, -1)
                        query_output = self.Qformer.bert(
                            query_embeds=query_tokens,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            return_dict=True,
                        )
                        image_embeds = self.video_proj(query_output.last_hidden_state)
                
                        _, image_embeds_length, image_embeds_hidden = image_embeds.size()
                        image_embeds = image_embeds.view(1, frame, image_embeds_length, image_embeds_hidden)
                        image_embeds = image_embeds.view(1, frame*image_embeds_length, image_embeds_hidden)

                    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.bool).to(image.device)

                if "prompt" in samples.keys():
                    prompt = samples["prompt"]
                else:
                    prompt = samples["text_input"]

                if use_image:
                    image_prompt = '<Image><ImageHere></Image>'
                    if use_video:
                        image_prompt = '<Video><VideoHere></Video>'
                    image_embeds, image_atts = self.prompt_wrap(image_embeds, image_atts, image_prompt, use_speech=False)


                device_type = "cuda" if "cuda" in str(self.device) else "cpu"
                opt_tokens = self.opt_tokenizer([prompt], return_tensors="pt", padding=True).to(self.device)
                opt_tokens = opt_tokens.to(self.device)
                context_length = opt_tokens.input_ids.size(1)    
                
                with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                    if use_image or use_video:
                        empty_targets = (
                            torch.ones(image_atts.size(), dtype=torch.long).to(image.device).fill_(0)
                        )
                        opt_tokens['input_ids'] = torch.cat([empty_targets, opt_tokens.input_ids], dim=1)
                        # opt_tokens['attention_mask'] = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
                        opt_tokens = opt_tokens.to(image.device)
                    
                        del opt_tokens['attention_mask']
                        del opt_tokens['position_ids']
                        context_length = opt_tokens.input_ids.size(1)
                        print('context_length: ', context_length)
                        # outputs = self.opt_model.generate(
                        #         **opt_tokens, **gen_kwargs, inputs_opt=inputs_opt,
                        #     )
                        for outputs in self.opt_model.stream_generate(**opt_tokens, **gen_kwargs, input_image=image_embeds,):
                            outputs = outputs.tolist()[0][context_length -  2:]
                            response = self.opt_tokenizer.decode(outputs)
                            response = self.process_response(response)
                        
                            yield response
                    else:
                        # outputs = self.opt_model.generate(
                        #     **opt_tokens, **gen_kwargs, 
                        # )
                        for outputs in self.opt_model.stream_generate(**opt_tokens, **gen_kwargs):
                            outputs = outputs.tolist()[0][context_length -  2:]
                            response = self.opt_tokenizer.decode(outputs)
                            response = self.process_response(response)
                        
                            yield response

    def process_response(self, response):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        response = response.replace("ChatGLM-6B", "百灵（Lark）")
        response = response.replace("ChatGLM", "百灵（Lark）")
        response = response.replace("清华大学 KEG 实验室和智谱 AI 公司", "中科院自动化所认知计算小组")
        response = response.replace("清华大学 KEG 实验室", "中科院自动化所")
        response = response.replace("智谱 AI 公司", "认知计算小组")
        response = response.replace("中科院自动化所和智谱AI", "中科院自动化所")
        response = response.replace("智谱AI", "")
        response = response.replace("清华大学KEG实验室和公司", "中科院自动化所")
        
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response
            


    @classmethod
    def from_config(cls, cfg):

        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        # Added by Minglun Han
        forward_mode = cfg.get("forward_mode", "")  # recently added by Minglun Han
        split_tasks = cfg.get("split_tasks", "")  # recently added by Minglun Han
        asr_loss_scale = cfg.get("asr_loss_scale", 0.0) # recently added by Minglun Han
        vd_loss_scale = cfg.get("vd_loss_scale", 0.0)   # recently added by Minglun Han
        vsd_loss_scale = cfg.get("vsd_loss_scale", 0.0) # recently added by Minglun Han
        speech_adaptor_type = cfg.get("speech_adaptor_type", "fc")

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            forward_mode=forward_mode,      # recently added by Minglun Han
            split_tasks=split_tasks,
            asr_loss_scale=asr_loss_scale,  # recently added by Minglun Han
            vd_loss_scale=vd_loss_scale,    # recently added by Minglun Han
            vsd_loss_scale=vsd_loss_scale,  # recently added by Minglun Han
            speech_adaptor_type=speech_adaptor_type,
        )

        model.load_checkpoint_from_config(cfg)

        # return model
    
        load_speech_adaptor = cfg.get("load_speech_adaptor", False)
        if load_speech_adaptor:
            logging.info(f"Load speech adaptor {speech_adaptor_type}")
            model.load_speech_adaptor_from_config(cfg)

        return model

    def load_speech_adaptor_from_config(self, cfg, **kwargs):
        load_speech_adaptor = cfg.get("load_speech_adaptor", True)
        if load_speech_adaptor:
            speech_adaptor_ckpt = cfg.get("speech_adaptor_ckpt", None)
            self.load_checkpoint(url_or_filename=speech_adaptor_ckpt, **kwargs)


    def preprocess_function_train(self, examples, device, src_txt_len=None, max_tgt_len=None):
        if src_txt_len is None:
            src_txt_len = self.max_txt_len
        if src_txt_len is None:
            max_tgt_len = self.max_txt_len

        max_seq_length = src_txt_len + max_tgt_len

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for question, answer in zip(examples["text_input"], examples["text_output"]):
            
            prompt = question
            a_ids = self.opt_tokenizer.encode(text=prompt, add_special_tokens=False)
            b_ids = self.opt_tokenizer.encode(text=answer, add_special_tokens=False)

            if len(a_ids) > src_txt_len - 1:
                a_ids = a_ids[: src_txt_len - 1]

            if len(b_ids) > max_tgt_len- 2:
                b_ids = b_ids[: max_tgt_len - 2]

            input_ids = self.opt_tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(self.opt_tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position+1:]
            
            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [self.opt_tokenizer.pad_token_id] * pad_len
            labels = labels + [self.opt_tokenizer.pad_token_id] * pad_len
            if self.ignore_pad_token_for_loss:
                labels = [(l if l != self.opt_tokenizer.pad_token_id else -100) for l in labels]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        model_inputs["input_ids"] = torch.LongTensor(model_inputs["input_ids"]).to(device)
        model_inputs["labels"] = torch.LongTensor(model_inputs["labels"]).to(device)
        return model_inputs
