import torch
import gc
import os
import deepspeed
import json
import re
import numpy as np
from tqdm import tqdm

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from datasets import load_dataset
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM)
from peft import (LoraConfig,
                  get_peft_model,
                  PeftModel,
                  TaskType)

MODEL_ID = "meta-llama/Llama-3.1-8B"
BENCHMARK_ID = "TUKE-DeutscheTelekom/skquad"

def flush():
    """
    Frees up unused GPU memory and resets memory statistics. It helps OOM errors by explicitly collecting garbage to free Python memory, 
    clearing the CUDA cache, collecting inter-process resources, and resetting peak memory usage statistics.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.ipc_collect()


def reformat_answer(row):
    texts = row["answers"]["text"]
    row["answers"] = texts[0]
    return row
    

def load_benchmark():
    benchmark = load_dataset(BENCHMARK_ID)
    filtered_benchmark = benchmark["train"].filter(lambda row: row["answers"]["text"])
    processed_benchmark = filtered_benchmark.map(reformat_answer)

    return processed_benchmark


def initialize_adapter(lora_r, lora_dropout, lora_alpha):
    """
    Initializes LoRA adapter configuration based on the currently set hyperparameters. The LoRA is applied to projection layers
    related to attention mechanism and feed-forward networks.

    Args:
        model: The base Llama 3.1 8B model to which LoRA adapters will be applied.
        lora_r (int): Rank for LoRA layers.
        lora_dropout (float): Dropout probability for LoRA layers.
        lora_alpha (float): Scaling factor for LoRA layers.

    Returns:
        LoraConfig: Configured LoRA adapter ready for integration with the model.
    Raises:
        ValueError: If any of the inputs are invalid.
    """

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


def response_correctness(response, ground_truth):
    cleaned_response = re.sub(r'[^A-Za-z0-9 ]', '', response)
    cleaned_ground_truth = re.sub(r'[^A-Za-z0-9 ]', '', ground_truth)
    
    response_words = cleaned_response.lower().split()
    ground_truth_words = set(cleaned_ground_truth.lower().split())
    return sum(1 for word in response_words if word in ground_truth_words and len(word) > 2)
    

def evaluate_benchmark(lora_output_path, benchmark):
    """
    Evaluates the fine-tuned model by generating text using a sample input.

    Args:
        lora_output_path (str): Path to the saved LoRA adapter.
        benchmark: Benchmark for the evaluation purpose
    Raises:
        ValueError: If the LoRA checkpoint path is invalid or missing.
        FileNotFoundError: If any required files are not found.
    """
    # validate LoRA output path
    if not os.path.exists(lora_output_path):
        raise FileNotFoundError(f"LoRA checkpoint not found at {lora_output_path}")
    
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        # load the tokenizer, base model of Llama 3.1 8B and the saved state dictionary of the model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        
        # load hyperparameters of the LoRA adapter
        try:
            lora_r, lora_alpha, lora_dropout = load_adapter(lora_output_path)
        except (FileNotFoundError, ValueError):
            raise ValueError(f"Not able to load LoRA from the checkpoint {lora_output_path}")

        # load state of the LoRA adapter from the saved checkpoint and interpret model as the PEFT model
        try:
            lora_config = initialize_adapter(lora_r, lora_dropout, lora_alpha)
        except ValueError:
            raise ValueError("Unable to unitialize LoRA adapter")
            
        model = get_peft_model(base_model, lora_config)
        peft_model = PeftModel.from_pretrained(model, lora_output_path)

        response_count = 0
        step = 0
        for question in tqdm(benchmark):
            if len(question["answers"].split()) > 5:
                continue
            input_text = f"{question["context"]} {question["question"]}"
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
            output = peft_model.generate(input_ids, max_new_tokens=50)
            
            generated_ids = output[0][input_ids.shape[-1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            correctness = response_correctness(response, question["answers"])
            if correctness:
                response_count += 1
            print(f"\nQuestion n.o. {question["id"]}:\nResponse: {response}\nAnswer: {question["answers"]}\nCorrectness: {correctness}")
                
            step += 1

        print(response_count / step * 100)
        # delete all the artefacts of the evaluation and free memory
        del tokenizer, base_model, model, peft_model
        flush()


if __name__ == "__main__":
    flush()
    benchmark = load_benchmark()
    evaluate_benchmark("adapters/1745647580", benchmark)
