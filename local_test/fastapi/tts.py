# pip install git+https://github.com/myshell-ai/MeloTTS.git --no-deps
from melo.api import TTS as MTTS

class TTS:
    def __init__(self):
        self.model = None
        self.speaker_ids = None
    
    def load(self):
        self.model = MTTS(language='ZH', device='cuda:0')
        self.speaker_ids = self.model.hps.data.spk2id
    
    def infer(self, text: str, output_path: str = './tmp/output_audio.wav', speed: float = 1.0):
        self.model.tts_to_file(text, self.speaker_ids['ZH'], output_path=output_path, speed=speed)

if __name__ == "__main__":
    tts = TTS()
    tts.load()
    tts.infer("If you can")