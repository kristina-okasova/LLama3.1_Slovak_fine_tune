import numpy as np
import torch
import gc
import os
import json
import keyboard

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM, LlamaForCausalLM)
from peft import (LoraConfig,
                  get_peft_model,
                  PeftModel, TaskType)

MODEL_ID = "meta-llama/Llama-3.1-8B"


def flush():
    """
    Frees up unused GPU memory and resets memory statistics. It helps OOM errors by explicitly collecting garbage to
    free Python memory, clearing the CUDA cache, collecting inter-process resources, and resetting peak memory usage
    statistics.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.ipc_collect()


def initialize_adapter(base_model, lora_r, lora_dropout, lora_alpha):
    """
    Initializes LoRA adapter configuration based on the currently set hyperparameters. The LoRA is applied to
    projection layers related to attention mechanism and feed-forward networks.

    Args:
        base_model: The base Llama 3.1 8B model to which LoRA adapters will be applied.
        lora_r (int): Rank for LoRA layers.
        lora_dropout (float): Dropout probability for LoRA layers.
        lora_alpha (float): Scaling factor for LoRA layers.

    Returns:
        LoraConfig: Configured LoRA adapter ready for integration with the model.
    Raises:
        ValueError: If any of the inputs are invalid.
    """
    # validate model type
    if not isinstance(base_model, LlamaForCausalLM):
        raise ValueError("The model must be an instance of AutoModelForCausalLM.")

    # validate lora_r
    if not (isinstance(lora_r, np.int64) or isinstance(lora_r, int)) or lora_r <= 0:
        raise ValueError("The 'lora_r' must be a positive integer.")
    
    # validate lora_dropout
    if not isinstance(lora_dropout, float) or not (0 <= lora_dropout <= 1):
        raise ValueError("The 'lora_dropout' must be a float between 0 and 1.")
    
    # validate lora_alpha
    if not (isinstance(lora_r, np.int64) or isinstance(lora_r, int)) or lora_alpha <= 0:
        raise ValueError("The 'lora_alpha' must be a positive float.")

    lora_config = LoraConfig(
        init_lora_weights=True,
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        r=int(lora_r),
        target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM
    )
    return lora_config


def load_adapter(lora_checkpoint_path):
    """
    Loads the LoRA adapter configuration from a checkpoint file.

    Args:
        lora_checkpoint_path (str): Path to the LoRA checkpoint directory.

    Returns:
        tuple: A tuple containing the LoRA adapter hyperparameters:
            - lora_r (int)
            - lora_alpha (float)
            - lora_dropout (float)
    Raises:
        FileNotFoundError: If the LoRA checkpoint directory or file does not exist.
        ValueError: If the hyperparameters in the JSON file are not valid.
    """
    # check if the LoRA checkpoint directory exists
    if not os.path.isdir(lora_checkpoint_path):
        raise FileNotFoundError(f"The directory {lora_checkpoint_path} does not exist.")
    
    # check if the LoRA checkpoint file exists
    adapter_config_file = f"{lora_checkpoint_path}/adapter_config.json"
    if not os.path.isfile(adapter_config_file):
        raise FileNotFoundError(f"The file {adapter_config_file} does not exist.")
    
    # Load the LoRA adapter configuration
    with open(adapter_config_file, "r") as file:
        try:
            adapter_config = json.load(file)
        except json.JSONDecodeError:
            raise ValueError(f"The file {adapter_config_file} is not a valid JSON file.")

    # get values of the configuration from dictionary
    try:
        lora_r = adapter_config["r"]
        lora_alpha = adapter_config["lora_alpha"]
        lora_dropout = adapter_config["lora_dropout"]
    except KeyError as e:
        raise ValueError(f"Missing key in adapter configuration: {e}")

    return lora_r, lora_alpha, lora_dropout


def load_model_tokenizer(model_output_path, lora_output_path):
    """
    Loads the fine-tuned model from the specified checkpoint path.

    Args:
        model_output_path (str): Path to the saved model checkpoint.
        lora_output_path (str): Path to the saved LoRA adapter.
    Raises:
        ValueError: If the model or LoRA checkpoint paths are invalid or missing.
        FileNotFoundError: If any required files are not found.
    """
    # validate model output path
    if not os.path.exists(model_output_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_output_path}")

    # validate LoRA output path
    if not os.path.exists(lora_output_path):
        raise FileNotFoundError(f"LoRA checkpoint not found at {lora_output_path}")
    
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        # load the FP32 model state from the specified checkpoint
        fp32_model_state = get_fp32_state_dict_from_zero_checkpoint(model_output_path)

        # load the tokenizer, base model of Llama 3.1 8B and the saved state dictionary of the model
        base_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        base_model.load_state_dict(fp32_model_state, strict=False)

        # load hyperparameters of the LoRA adapter
        try:
            lora_r, lora_alpha, lora_dropout = load_adapter(lora_output_path)
        except (FileNotFoundError, ValueError):
            raise ValueError(f"Not able to load LoRA from the checkpoint {lora_output_path}")

        # load state of the LoRA adapter from the saved checkpoint and interpret model as the PEFT model
        try:
            lora_config = initialize_adapter(base_model, lora_r, lora_dropout, lora_alpha)
        except ValueError:
            raise ValueError("Unable to initialize LoRA adapter")
            
        lora_model = get_peft_model(base_model, lora_config)
        peft_model = PeftModel.from_pretrained(lora_model, lora_output_path)

        return peft_model, base_tokenizer


def inference(inference_model=None, inference_tokenizer=None):
    """
    Inference mode for the fine-tuned version of the Llama 3.1 8B. The texts are generated based on the user's input.

    Args:
        inference_model: the fine-tuned model loaded from the checkpoints
        inference_tokenizer: loaded tokenizer of Llama 3.1
    """
    if inference_model is None or inference_tokenizer is None:
        print("Could not load model or tokenizer")
        return

    print("INFERENCE MODE:\nType input and the fine-tuned version of Llama 3.1 8B will generate the continuation of "
          "the text you provided. Try to challenge the model's abilities in Slovak language.")
    
    while True:
        # wait for user input
        user_input = input("Enter something: ")

        # generate text based on input
        input_ids = inference_tokenizer(user_input, return_tensors="pt").input_ids.to(inference_model.device)
        output = inference_model.generate(input_ids, max_new_tokens=50)
        print(f"GENERATED TEXT:\n{inference_tokenizer.decode(output[0], skip_special_tokens=True)}\n")

        # check if ESC key is pressed to exit the loop
        if keyboard.is_pressed('esc'):
            print("Stopping the inference mode...")
            break

        # delete all the artefacts of the inference and free memory
        del inference_tokenizer, inference_model
        flush()
    

if __name__ == "__main__":
    flush()
    model, tokenizer = load_model_tokenizer("", "")
    inference(model, tokenizer)
