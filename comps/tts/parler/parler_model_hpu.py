# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import numpy as np
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu 

class ParlerTTSModel:
    def __init__(self) -> None:
        pass

    def t2s(self, text: str) -> None:
        return None
    
if __name__ == "__main__":
    device = "hpu" if hthpu.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1").to(device) # Aug. released model
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

    # prompt = "Hey, how are you doing today?"
    # prompt = "The Open Platform for Enterprise AI (OPEA) is a collaborative initiative that aims to accelerate the adoption and development of Artificial intelligence (AI) in enterprises. It provides a standardized framework, tools and resources to facilitate the integration of AI technologies into various business processes."
    prompt = "The Open Platform for Enterprise Artificial Intelligence (OPEA) is a collaborative initiative that aims to accelerate the adoption and development of Artificial intelligence in enterprises. It provides a standardized framework, tools and resources to facilitate the integration of artificial intelligence technologies into various business processes."

    description = "A male speaker with a slightly high-pitched voice delivering his words at a moderate pace in a small, confined space with a touch of background noise and a quite monotone tone."

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)

    # # Tranform the audio to base64 string
    # sf.write("tmp.wav", all_speech, samplerate=16000)
    # with open("tmp.wav", "rb") as f:
    #     bytes = f.read()
    # import base64

    # b64_str = base64.b64encode(bytes).decode()
    # assert b64_str[:3] == "Ukl"


    
