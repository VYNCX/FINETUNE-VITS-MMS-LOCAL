# Finetune VITS and MMS on Local's tools

This repo is original by : https://github.com/ylacombe/finetune-hf-vits

## 1. Requirements

```sh
git clone https://github.com/VYNCX/FINETUNE-VITS-MMS-LOCAL.git
cd finetune-hf-vits
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

For example Thai language use : tha
All language support : [Check MMS Language Support](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html)

```sh
cd <path-to-finetune-hf-vits-repo>
python convert_original_discriminator_checkpoint.py --language_code tha --pytorch_dump_folder_path <local-folder>
```
