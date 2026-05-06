# AV_Projects
This repository contains various projects related to ML, Computer Vision and NLP

# Vision-Language Assistant using MobileVLM

A lightweight multimodal AI assistant capable of understanding images and generating contextual natural language responses using a fine-tuned Mobile Vision-Language Model (MobileVLM).

This project combines a CLIP-based vision encoder with a MobileLLaMA language model and uses parameter-efficient fine-tuning (LoRA) to build an efficient assistant-style multimodal system suitable for mobile and edge-assisted applications.

Features
- Multimodal image + text understanding
- Context-aware natural language generation
- Fine-tuned MobileLLaMA (1.4B) using LoRA
- CLIP vision encoder integration
- Learned visual-language projection layer (LDP projector)
- Reduced hallucinations through prompt engineering and training strategy
- Optimized memory usage with PEFT (Parameter-Efficient Fine-Tuning)
- Azure VM deployment with REST API inference service
- Mobile backend integration

 ## Architecture Overview

                 +------------------+
                |   Input Image    |
                +------------------+
                          |
                          v
                +------------------+
                |  CLIP Encoder    |
                +------------------+
                          |
                Visual Embeddings
                          |
                          v
                +------------------+
                |   LDP Projector  |
                +------------------+
                          |
             Projected Visual Tokens
                          |
                          v
      +-----------------------------------+
      |  MobileLLaMA (1.4B) + LoRA        |
      |  Multimodal Fusion + Generation   |
      +-----------------------------------+
                          |
                          v
                Natural Language Response


## Tech Stack
1. Models
- MobileLLaMA (1.4B)
- CLIP Vision Encoder
- LoRA / PEFT

2. Frameworks
- PyTorch
- Hugging Face Transformers
- PEFT
- Accelerate

3. Deployment
- Azure VM
- FastAPI / Flask API

4. Dataset Used
   Flickr8k Used for:
   - Image-caption grounding
   - Visual semantic understanding
   - Caption alignment
   
   LLaVA-Instruct Used for:
   - Instruction tuning
   - Assistant-style reasoning
   - Multimodal conversational learning

5. Fine-Tuning Strategy

The project uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters to reduce GPU memory usage and training cost.
LoRA applied on:
- Query projection layers
- Key projection layers
- Value projection layers
- Output projection layers
