from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf
import numpy as np

class SpeechToTextImpaired:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        self.device = torch.device("cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def transcribe(self, audio_file_path):
        # Load audio
        speech, sr = sf.read(audio_file_path)
        if sr != 16000:
            # Resample to 16kHz (librosa)
            import librosa
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000
        # Process
        inputs = self.processor(speech, sampling_rate=sr, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.lower()