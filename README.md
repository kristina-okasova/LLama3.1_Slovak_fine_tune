# Fine-tuned version of Llama 3.1 8B for Slovak language comprehension 

The fine-tuned version of Llama 3.1 8B with improved Slovak language comprehension abilities. The model is available in inference mode with command-line user interface. In inference mode, the model is able to generate text in Slovak language covering facts from various fields of information related to Slovakia. For now, just the fine-tuned version of the base model is available, considering instruct version as future work.

Moreover, the script covering the whole fine-tuning process is available covering loading of Slovak datasets (TO-DO link), their tokenization using Llama 3.1's tokenizer, loading of the model with utilization of PEFT techniques, specifically 4-bit quantization and LoRA adapter. The fine-tuning process is optimized for distributed environment, whose parameters are set via DeepSpeed configuration. The training loop is followed by validation with calculation of perplexity metric. All metrics, hyperparameters and progress of loss function are recorded via Wandb. The final fine-tuned model is saved in format of DeepSpeed checkpoint.

Further, multiple Slovak tokenizers were trained and are available as well, namely BPE, WordPiece, Unigram and SentencePiece. However, they are not being used for the tokenization of the datasets due to problems of saving the checkpoints after resizing the embedding sizes of the model, that is a requirement when adjusting the vocabulary of the tokenizer of the base model.

## Requirements
1. User's account on Hugging Face
2. Approved access to Llama 3.1 (https://huggingface.co/meta-llama/Llama-3.1-8B)
3. Python version TO-DO or higher
4. Required libraries: TO-DO
5. Sufficient hardware - at least 2x ASUS GeForce RTX 4090 (only necessary for fine-tuning)
6. User's account on Wandb (only necessary for fine-tuning)
7. Specification of the path to the datasets - due to size limits, not able to store them as part of the repository (only necessary for fine-tuning)

## Inference mode
In order to use the fine-tuned model, run `python3 inference_mode.py`. The user will be asked to insert any text that should be completed by the generated text from the model. After displaying the generated answer, the user is asked to provide next input, until Esc key is pressed. The time required for generating the text differs based on the hardware setup (access to 1 GPU could be required even for inference mode).

## Fine-tuning mode
When aiming to reproduce the fine-tuning process of the Llama 3.1 8B model, the whole repository has to be cloned to the local machine to preserve the folder structure required by the model, that is desribed below. To clone the repository run `git clone https://github.com/kristina-okasova/LLama3.1_Slovak_fine_tune.git`.

## Repository structure
