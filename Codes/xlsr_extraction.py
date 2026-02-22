import os
import sys
import numpy as np
import pandas as pd
import librosa
import torch
import warnings

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from torch.cuda.amp import autocast

# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

# Check GPU availability
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    multi_gpu = True
else:
    print("Using single GPU or CPU")
    multi_gpu = False

# Load processor and model
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Multi-GPU wrapping
if multi_gpu:
    model = torch.nn.DataParallel(base_model)
else:
    model = base_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Paths
input_path = "/input_path_to_audio_samples"
output_path = "/output_path_to_save_the_represntations"
excel_file_path = os.path.join(output_path, "audio_filenames_xlsr.xlsx")
output_npy_path = os.path.join(output_path, "xlsr_hidden_states.npy")
os.makedirs(output_path, exist_ok=True)

# Constants
target_length = 16000 * 11

def process_audio_file(file_name, audio_path):
    waveform, _ = librosa.load(audio_path, sr=16000)
    waveform = torch.tensor(waveform).unsqueeze(0)  # Add batch dimension

    # Pad or truncate
    if waveform.shape[1] < target_length:
        pad_length = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))
    else:
        waveform = waveform[:, :target_length]

    waveform = waveform.to(device)
    inputs = processor(waveform.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        with autocast(dtype=torch.float16):
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state

    return hidden_states.squeeze().cpu().numpy()

# Collect and sort files
all_files = sorted([f for f in os.listdir(input_path) if f.endswith(".wav")]) 
audio_filenames = []

for idx, file_name in enumerate(all_files[0:844], start=0):  # choose how many files you wan to be processed 
    print(f"Processing file {idx}: {file_name}")
    audio_path = os.path.join(input_path, file_name)
    hidden_states = process_audio_file(file_name, audio_path)

    # Save to .npy
    if not os.path.exists(output_npy_path) or os.stat(output_npy_path).st_size == 0:
        np.save(output_npy_path, np.array([hidden_states]))
    else:
        existing_data = np.load(output_npy_path, allow_pickle=True)
        new_data = np.append(existing_data, [hidden_states], axis=0)
        np.save(output_npy_path, new_data)

    audio_filenames.append(file_name)
    torch.cuda.empty_cache()

# Save filenames
df = pd.DataFrame({"audio_filenames": audio_filenames})
df.to_excel(excel_file_path, index=False)

print(f"✅ Processing complete! Output saved to:\n- {output_npy_path}\n- {excel_file_path}")


