import os
import torchaudio
from tqdm import tqdm
import soundfile as sf

# Define constants
AUDIO_DIR = "./dataset/BANK_TH/audio"
OUTPUT_DIR = "./dataset/BANK_TH/preprocessed_audio"
TARGET_SAMPLE_RATE = 22050
TARGET_FORMAT = "wav"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_audio_file(input_path, output_path, target_sample_rate):
    """
    Preprocess a single audio file:
    - Convert to mono channel.
    - Resample to the target sample rate.
    - Save in the specified format.
    """
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(input_path)
        
        # Convert to mono if not already
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # Save the processed audio
        torchaudio.save(output_path, waveform, sample_rate=target_sample_rate)
        return True
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")
        return False

def preprocess_audio_dataset(audio_dir, output_dir, target_sample_rate, target_format):
    """
    Preprocess all audio files in the dataset directory.
    """
    print(f"Processing audio files in {audio_dir}...")

    processed_files = 0
    failed_files = 0

    for file in tqdm(os.listdir(audio_dir), desc="Preprocessing Audio Files"):
        input_path = os.path.join(audio_dir, file)
        output_filename = os.path.splitext(file)[0] + f".{target_format}"
        output_path = os.path.join(output_dir, output_filename)

        # Process each audio file
        if preprocess_audio_file(input_path, output_path, target_sample_rate):
            processed_files += 1
        else:
            failed_files += 1

    print(f"Completed preprocessing: {processed_files} files processed, {failed_files} files failed.")
    print(f"Processed files saved to {output_dir}.")

# Run preprocessing
preprocess_audio_dataset(AUDIO_DIR, OUTPUT_DIR, TARGET_SAMPLE_RATE, TARGET_FORMAT)
