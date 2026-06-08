import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import base64
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # for text-to-speech placeholder

class SignCNN(nn.Module):
    """Simple CNN for hand gesture recognition (e.g., ISL alphabet)."""
    def __init__(self, num_classes=36):  # 26 letters + 10 digits
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)   # assuming input 224x224
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SignLanguageRecognizer:
    def __init__(self, weights_path=None):
        self.device = torch.device("cpu")
        self.num_classes = 36
        self.model = SignCNN(self.num_classes).to(self.device)
        # Load pretrained weights if available, else use random (dummy)
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            warnings.warn("No sign language weights provided. Using untrained model (dummy outputs).")
        self.model.eval()
        self.labels = [chr(ord('A')+i) for i in range(26)] + [str(i) for i in range(10)]
        # Dummy TTS (use gTTS or pyttsx3; here we simulate base64 audio)
        try:
            from gtts import gTTS
            import io
            self.tts_available = True
        except:
            self.tts_available = False

    def preprocess(self, frame):
        """Resize 224x224, normalize, convert to tensor."""
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img

    def predict(self, frame):
        img_tensor = self.preprocess(frame)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()
        return self.labels[pred_idx]

    def text_to_speech(self, text):
        """Return base64 encoded audio (WAV/MP3)."""
        if self.tts_available:
            from gtts import gTTS
            import io
            tts = gTTS(text=text, lang='hi' if any(ord(c) > 127 for c in text) else 'en', slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return base64.b64encode(fp.read()).decode()
        else:
            # Dummy silence (1 sec)
            import soundfile as sf
            import numpy as np
            dummy_audio = np.zeros(16000, dtype=np.float32)
            fp = io.BytesIO()
            sf.write(fp, dummy_audio, 16000, format='wav')
            fp.seek(0)
            return base64.b64encode(fp.read()).decode()