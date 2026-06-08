from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import base64
import torch

class SceneDescriber:
    def __init__(self):
        self.device = torch.device("cpu")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        self.model.eval()
        # TTS same as above
        try:
            from gtts import gTTS
            self.tts_available = True
        except:
            self.tts_available = False

    def describe(self, image):
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=50)
        description = self.processor.decode(out[0], skip_special_tokens=True)
        return description

    def text_to_speech(self, text):
        # Same as in sign_language.py
        if self.tts_available:
            from gtts import gTTS
            import io
            tts = gTTS(text=text, lang='en', slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return base64.b64encode(fp.read()).decode()
        else:
            import soundfile as sf
            import numpy as np
            dummy_audio = np.zeros(16000, dtype=np.float32)
            fp = io.BytesIO()
            sf.write(fp, dummy_audio, 16000, format='wav')
            fp.seek(0)
            return base64.b64encode(fp.read()).decode()