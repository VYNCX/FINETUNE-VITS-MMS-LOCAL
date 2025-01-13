# Finetune VITS and MMS on Local

Original Repo : https://github.com/ylacombe/finetune-hf-vits

## 1. Requirements

```sh
git clone https://github.com/VYNCX/finetune-local-vits.git
cd finetune-local-vits
pip install -r requirements.txt
#for thai language
pip install pythainlp
```

Build the monotonic alignment search function using cython. This is absolutely necessary since the Python-native-version is awfully slow.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
cd ..
```
## 2. Download Pretrained model

For example Thai language use : tha ,
All language support : [Check MMS Language Support](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html)

```sh
cd finetune-local-vits
python convert_original_discriminator_checkpoint.py --language_code tha --pytorch_dump_folder_path <local-folder> #example ./models_dump
```
## 3. Prepare Dataset and Config file

to prepare dataset in dataset folder or your path.its support for 
    3-10 Sec per audio clip, Naturalness recordings.
    16000-22050 Sample-rate for audio (MMS pretrained model used 16kHz)

**Example** 
```text
/dataset
 - metadata.csv
 - /audio-data
    - /train
        - /audio1.wav
```
Metadata.csv

```text
file_name,text
audio-data/train/audio1.wav,สวัสดีครับทุกคน ยินดีที่ได้พบกันอีกครั้ง
audio-data/train/audio2.wav,เธอเคยเห็นนกบินสูงบนฟ้าสีครามไหม
```
You can prepare config .json file in **training_config_examples** directory. Remember name of .json file and directory for finetuning method.
**Example** :
```json
{
    "project_name": "your_project_name",
    "push_to_hub": false,
    "hub_model_id": "",
    "report_to": ["tensorboard"], <-- remove if you don't want to virtualize train process.
    "overwrite_output_dir": true,
    "output_dir": "your_output_directory", <-- your output directory "./output" for local.

    "dataset_name": "./dataset", <-- your dataset directory "./mms-tts-datasets/train" for local.
    "audio_column_name": "audio",
    "text_column_name": "text",
    "train_split_name": "train",
    "eval_split_name": "train",

    "full_generation_sample_text": "ในวันหยุดสุดสัปดาห์ การไปช็อปปิ้งที่ห้างสรรพสินค้าเป็นกิจกรรมที่ทำให้เราผ่อนคลายจากความเครียดในชีวิตประจำวัน",

    "max_duration_in_seconds": 20,
    "min_duration_in_seconds": 1.0,
    "max_tokens_length": 500,

    "model_name_or_path": "your_model_path_for_pretrained_model", <-- this model from "Download Pretrained model" method.

    "preprocessing_num_workers": 4,

    "do_train": true,
    "num_train_epochs": 200,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": false,
    "per_device_train_batch_size": 8, <-- decrease this parameter if you have less VRAM.
    "learning_rate": 2e-5,
    "adam_beta1": 0.8,
    "adam_beta2": 0.99,
    "warmup_ratio": 0.01,
    "group_by_length": false,

    "do_eval": true,
    "eval_steps": 50,
    "per_device_eval_batch_size": 8, <-- decrease this parameter if you have less VRAM.
    "max_eval_samples": 20, <-- increase this parameter if you have less sample audio.
    "do_step_schedule_per_epoch": true,

    "weight_disc": 3,
    "weight_fmaps": 1,
    "weight_gen": 1,
    "weight_kl": 1.5,
    "weight_duration": 1,
    "weight_mel": 35,

    "fp16": true,
    "seed": 456
}
```


## 4. Finetuning

There are two ways to run the finetuning scrip, both using command lines. Note that you only need one GPU to finetune VITS/MMS as the models are really lightweight (83M parameters).
**Need to prepare config file before finetuning.**

```sh
accelerate launch run_vits_finetuning.py ./training_config_examples/finetune_mms_thai.json
```

## 5. inference

**Run** :

```python
from transformers import pipeline
import scipy

model_id = "modelpath or huggingface model" #your trained model path
synthesiser = pipeline("text-to-speech", model_id) # add device=0 if you want to use a GPU

speech = synthesiser("สวัสดีครับ นี่คือเสียงพูดภาษาไทย") #your text here

scipy.io.wavfile.write("finetuned_output.wav", rate=speech["sampling_rate"], data=speech["audio"][0])
```

or use with Sample Gradio : 

```sh
python inference-gradio.py
```
