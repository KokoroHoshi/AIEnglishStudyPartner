from pathlib import Path
from typing import Optional
from PIL import Image
from io import BytesIO

import requests
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig

from relationalDB import RelationalDB

import uuid
from datetime import datetime, timezone

class VLM:
    def __init__(self, model_id: str, cache_dir: str, hf_token: Optional[str] = None, db_instance: RelationalDB = None):
        self.id = model_id
        self.cache_dir = Path(cache_dir)
        self.hf_token = hf_token
        self.model = None
        self.processor = None
        self.db = db_instance

    def load(self):
        if self.hf_token is None: 
            print("Access to model google/paligemma-3b-mix-224 is restricted. You must be authenticated to access it.")
            print("hf token is needed")
            return 
        
        vlm_quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.id, 
            quantization_config=vlm_quantization_config,
            low_cpu_mem_usage=True, 
            cache_dir=str(self.cache_dir),
            token=self.hf_token
        ).eval()

        self.processor = AutoProcessor.from_pretrained(self.id, cache_dir=str(self.cache_dir))
        print(f"{id(self)} VLM loaded successfully.")
    
    def unload(self):
        # if not self.model_loaded:
        #     print("model is not loaded.")
        #     return
        
        del self.model
        del self.processor

        torch.cuda.empty_cache()

        # self.model_loaded = False
        print(f"{id(self)} VLM has been unloaded successfully.")
    
    def infer(self, image_bytes: bytes, prompt: str = "What is shown in this image?") -> str:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        vlm_inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        vlm_input_len = vlm_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**vlm_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][vlm_input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
        
        return decoded

    def infer_by_url(self, image_url: str = "https://www.ilankelman.org/stopsigns/australia.jpg",
               prompt: str = "What is shown in this image?") -> str:
        image = Image.open(requests.get(image_url, stream=True).raw)
        vlm_inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        vlm_input_len = vlm_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**vlm_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][vlm_input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
        
        return decoded

    def infer_with_db(self, user_id: str, image_bytes: bytes, prompt: str = "What is shown in this image?", add_to_history: bool = True) -> str:
        if self.db is None:
            raise ValueError("Database instance is not set for VLM.")

        if add_to_history:
            cache_id = str(uuid.uuid4())
            cache_type = 'image'
            timestamp = datetime.now(timezone.utc).isoformat()

            self.db.insert_data(
                table_name='user_cache',
                columns='cache_id, cache_type, cache_data, timestamp',
                values=(cache_id, cache_type, image_bytes, timestamp)
            )
            self.db.insert_data(
                table_name='user_cache_relation',
                columns='user_id, cache_id',
                values=(user_id, cache_id)
            )

        return self.infer(image_bytes=image_bytes, prompt=prompt)

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')

    vlm_id = "google/paligemma-3b-mix-224"
    cache_dir = './models'

    vlm = VLM(vlm_id, cache_dir, HF_TOKEN)
    vlm.load()

