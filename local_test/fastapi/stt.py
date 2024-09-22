from typing import Optional
from pathlib import Path
from io import BytesIO
from pydub import AudioSegment


import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class STT:
    def __init__(self, model_id: str, cache_dir: str, hf_token: Optional[str] = None):
        self.id = model_id
        self.cache_dir = Path(cache_dir)
        self.hf_token = hf_token
        self.pipeline = None

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
    def save_m4a_bytes_to_wav(self, audio_bytes: BytesIO, output_file_path: str = "./tmp/user_audio.wav"):
        audio_io = BytesIO(audio_bytes)
        audio_segment = AudioSegment.from_file(audio_io, format="m4a")
        audio_segment.export(output_file_path, format="wav")

    def infer(self, file_path: str) -> str:
        result = self.pipeline(file_path)
        return result

if __name__ == "__main__":
    model_id = "openai/whisper-large-v3"
    cache_dir = './models'

    stt = STT(model_id, cache_dir)
    stt.load()

