from transformers import pipeline
import gradio
import scipy
import torch

def generate_speech(text, model_path):
    synthesiser = pipeline("text-to-speech", model_path, device=0 if torch.cuda.is_available() else -1)
    speech = synthesiser(text)
    
    # Resample to 48kHz if needed
    if speech["sampling_rate"] != 48000:
        resampled_audio = scipy.signal.resample(speech["audio"][0], int(len(speech["audio"][0]) * 48000 / speech["sampling_rate"]))
        sampling_rate = 48000
    else:
        resampled_audio = speech["audio"][0]
        sampling_rate = speech["sampling_rate"]
    
    return sampling_rate, resampled_audio

def save_audio(sampling_rate, audio_data, filename="output.wav"):
    scipy.io.wavfile.write(filename, rate=sampling_rate, data=audio_data)
    return filename

def ui_fn(text, model_path):
    sampling_rate, audio_data = generate_speech(text, model_path)
    audio_file = save_audio(sampling_rate, audio_data)
    return audio_file

if __name__ == "__main__":
    iface = gradio.Interface(
        fn=ui_fn,
        inputs=[
            gradio.Textbox(label="Text to Synthesize"),
            gradio.Textbox(label="Model Path", value="./models_thaiv2")
        ],
        outputs=gradio.Audio(label="Generated Audio"),
        title="Text-to-Speech Synthesizer",
        description="Enter text and model path to generate speech."
    )
    iface.launch()
