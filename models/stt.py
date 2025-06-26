from typing import Optional
from pathlib import Path
from io import BytesIO
from pydub import AudioSegment

import uuid
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from models.relationalDB import RelationalDB
from datetime import datetime, timezone

from os import remove

class STT:
    def __init__(self, model_id: str, cache_dir: str, hf_token: Optional[str] = None, db_instance: RelationalDB = None):
        self.id = model_id
        self.cache_dir = Path(cache_dir)
        self.hf_token = hf_token
        self.pipeline = None
        self.db = db_instance

    def load(self):
        stt = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            use_safetensors=True, 
            attn_implementation="sdpa",
            cache_dir=str(self.cache_dir)
        ).to('cuda:0')

        stt_processor = AutoProcessor.from_pretrained(self.id)

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=stt,
            tokenizer=stt_processor.tokenizer,
            feature_extractor=stt_processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype='torch.float16',
            device='cuda:0',
        )
        print(f"{id(self)} STT loaded successfully.")

    def unload(self):
        # if not self.model_loaded:
        #     print("model is not loaded.")
        #     return
        
        del self.pipeline

        torch.cuda.empty_cache()

        # self.model_loaded = False
        print(f"{id(self)} STT has been unloaded successfully.")
    
    # need to install ffmpeg first
    def save_m4a_bytes_to_temp_wav(self, audio_bytes: bytes, output_file_path: str = "./tmp/user_audio.wav"):
        audio_io = BytesIO(audio_bytes)
        audio_segment = AudioSegment.from_file(audio_io, format="m4a")
        audio_segment.export(output_file_path, format="wav")

    def infer(self, file_path: str) -> str:
        result = self.pipeline(file_path)
        return result["text"]
    
    async def infer_with_db(self, user_id: str, audio_bytes: bytes, add_to_history: bool = True) -> str:
        if add_to_history:
            cache_id = str(uuid.uuid4())
            cache_type = 'audio'
            timestamp = datetime.now(timezone.utc).isoformat()

            self.db.insert_data(
                'user_cache',
                'cache_id, cache_type, cache_data, timestamp',
                (cache_id, cache_type, audio_bytes, timestamp)
            )

            self.db.insert_data(
                table_name='user_cache_relation',
                columns='user_id, cache_id',
                values=(user_id, cache_id)
            )

        temp_path = f"./tmp/temp_user_audio_{uuid.uuid4().hex}.wav"
        self.save_m4a_bytes_to_temp_wav(audio_bytes, temp_path)

        result_text = self.infer(temp_path)

        # remove(temp_path)

        return result_text

if __name__ == "__main__":
    model_id = "openai/whisper-large-v3"
    cache_dir = './models'

    stt = STT(model_id, cache_dir)
    stt.load()

