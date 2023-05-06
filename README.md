# X-LLM
X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages

[[Project Page](https://x-llm.github.io/)] [[Paper]()]

X-LLM converts multi-modalities (images, speech, videos) into foreign languages using X2L interfaces and feed them into a large Language Model (ChatGLM) to accomplish a Multimodal LLM, achieving impressive multimodal chat capabilities.

X-LLM is a general multimodal LLM framework that allows us to incorporate various modalities of information into LLMs, such as (1) non-speech audios, enabling the LLM to have conversations about audios (2) terminal device status information, enabling LLM to control terminal devices, and so on.

X-LLM connects multiple pre-trained single-modal encoders (such as ViT-g visual encoder) and large language model ChatGLM, using X2L interfaces. We consider a three-stage training procedure:
- Stage 1: Converting Multimodal Information. Convert multimodal information into foreign languages through X2L interfaces, only X2L interfaces are updated
- Stage 2: Aligning X2L Representations with the LLM. Inject foreign languages into LLM, only X2L interfaces are updated.
- Stage 3: Integrating Multiple Modalities. Integrating multi-modalities, only the adapters in X2L interfaces are updated.
