# Fine-tuned version of Llama 3.1 8B for Slovak language comprehension 

The fine-tuned version of Llama 3.1 8B with improved Slovak language comprehension abilities. The model is available in inference mode with command-line user interface. In inference mode, the model is able to generate text in Slovak language covering facts from various fields of information related to Slovakia. For now, just the fine-tuned version of the base model is available, considering instruct version as future work.

Moreover, the script covering the whole fine-tuning process is available covering loading of Slovak datasets (TO-DO link), their tokenization using Llama 3.1's tokenizer, loading of the model with utilization of PEFT techniques, specifically 4-bit quantization and LoRA adapter. The fine-tuning process is optimized for distributed environment, whose parameters are set via DeepSpeed configuration. The training loop is followed by validation with calculation of perplexity metric. All metrics, hyperparameters and progress of loss function are recorded via Wandb. The final fine-tuned model is saved in format of DeepSpeed checkpoint.

Further, multiple Slovak tokenizers were trained and are available as well, namely BPE, WordPiece, Unigram and SentencePiece. However, they are not being used for the tokenization of the datasets due to problems of saving the checkpoints after resizing the embedding sizes of the model, that is a requirement when adjusting the vocabulary of the tokenizer of the base model.

## Requirements
1. User's account on Hugging Face
2. Approved access to Llama 3.1 (https://huggingface.co/meta-llama/Llama-3.1-8B)
3. Python version 3.12.5 or higher
4. Required libraries are listed in `requirements.txt`
6. Sufficient hardware - at least 2x ASUS GeForce RTX 4090 (only necessary for fine-tuning)
7. User's account on Wandb (only necessary for fine-tuning)
8. Specification of the path to the datasets - due to size limits, not able to store them as part of the repository (only necessary for fine-tuning)

## Inference mode
In order to use the fine-tuned model, run the following command.
```bash
python3 inference_mode.py
```
The user will be asked to insert any text that should be completed by the generated text from the model. After displaying the generated answer, the user is asked to provide next input, until Esc key is pressed. The time required for generating the text differs based on the hardware setup (access to 1 GPU could be required even for inference mode).

## Fine-tuning mode
When aiming to reproduce the fine-tuning process of the Llama 3.1 8B model, the whole repository has to be cloned to the local machine to preserve the folder structure required by the model, that is desribed below. To clone the repository run the following command.
```bash
git clone https://github.com/kristina-okasova/LLama3.1_Slovak_fine_tune.git
```
In order to initialize the fine-tuning process, specification of datasets is required, placing them to the folder `datasets`. Adjust the constants at the beginning to exactly meet your requirements. The private information (Hugging Face token, Wandb project and user name) have to be store in .env file with the following structure:
```text
HUGGING_FACE_TOKEN=hugging_face_token
WANDB_PROJECT_NAME=wandb_project_name
WANDB_USER_NAME=wandb_user_name
```
To start the fine-tuning process in the distributed environment run the following command with adjusted number of GPUs available.
```bash
deepspeed --num_gpus=number_of_gpus fine_tune_mode.py
```
As the execution of fine-tuning process requires vast amount of time, in Linux environment it is recommended to consider usage of `screen` command.
```bash
screen -S screen_name
deepspeed --num_gpus=number_of_gpus fine_tune_mode.py 2>&1 | tee console_output.txt
```
Press `Ctrl+a`, then `d` to detach from the screen and `screen -r` to reatach the screen.

The fine-tuning should be applicable on any open-source base Large Language Model that support 4-bit quantization and append of LoRA adapter (however, no model outside Llama 3 family was tested).

## Repository structure
The repository contains the following directory structure.
```text
.
├── Llama3.1_fine_tune
│   └── README.md
├── README.md
├── adapters
│   └── README.md
├── datasets
│   └── README.md
├── ds_config.json
├── evaluation_pipeline.py
├── fine_tune_mode.py
├── hpo_results
│   └── README.md
├── inference_mode.py
├── requirements.txt
└── tokenized_dataset
    └── README.md

5 directories, 11 files
```
The individual directories contain their own README.md files, where the purpose of the directory and the expected files to be stored inside is specified. The remaining files are described below:
1. `ds_config.json` - configuration file for DeepSpeed distributed environment
2. `evaluation_pipeline.py` - sequence of inputs for the evaluation and comparison of various versions of fine-tuned models
3. `fine_tune_mode.py` - the fine-tuning script
4. `inference_mode.py` - the script for interaction with the fine-tuned model
5. `requirements.txt` - the list of all requirements
