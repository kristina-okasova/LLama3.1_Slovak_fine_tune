# Importing libraries
import torch
import numpy as np
import wandb
import gc
import datasets
import os
import random
import skopt.plots
import json
import deepspeed
import time

from tqdm import tqdm
from torch.utils.data import DataLoader
from huggingface_hub import login
from datasets import Dataset
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from collections import deque
from datasets import DatasetDict
from dotenv import load_dotenv

from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          LlamaForCausalLM,
                          DataCollatorForLanguageModeling,
                          BitsAndBytesConfig,
                          get_scheduler)
from peft import (LoraConfig,
                  get_peft_model,
                  prepare_model_for_kbit_training,
                  PeftModel,
                  TaskType)
from skopt.space import (Integer, 
                         Real)

# Definition of constants
ATTENTION_MECHANISM = "flash_attention_2"
# SPECIFY YOUR OWN DATASET NAME
DATASET_NAME = "OPUS"
DATASET_DIRECTORY = "tokenized_dataset"
DEEPSPEED_CONFIG = "ds_config.json"
HPO_DIRECTORY = "hpo_results"
MAX_LOSS_VALUE = 10000
MAX_SEQUENCE_LENGTH = 1024
MODEL_ID = "meta-llama/Llama-3.1-8B"
MODEL_OUTPUT_DIRECTORY = "Llama3.1_fine_tune"
MODEL_PATH_RANK0 = "zero_pp_rank_0_mp_rank_00_model_states.pt"
MODEL_PATH_RANK1 = "zero_pp_rank_1_mp_rank_00_model_states.pt"
NUMBER_OF_HPO_CALLS = 10
NUMBER_OF_TRAINING_EPOCHS = 3
NUMBER_OF_TRAINING_STEPS = 1000
NUMBER_OF_WARM_UP_STEPS = 100
OPTIMIZER_NAME = "AdamW"
TRAIN_TEST_RATIO = 0.2

# Definition of space for hyperparameter tuning
space = [Real(1e-5, 1e-1, "log-uniform", name="learning_rate"),
         Real(0.01, 0.1, name="lora_dropout"),
         Integer(4, 128, name="lora_r"),
         Integer(8, 128, name="lora_alpha"),
         Real(0.05, 0.1, name="warm_up_ratio"),
         Real(1e-5, 1e-2, name="weight_decay")]


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


@skopt.utils.use_named_args(space)
def objective(**params):
    """
    Objective function for hyperparameter optimization with skopt that runs a training loop with current configuration
    of the hyperparameters.
    Args:
        **params: Dictionary containing hyperparameters for training.

    Returns:
        float: Average validation loss as evaluation metric used for optimization
    """
    flush()
    print(params)
    try:
        loss = train(params["lora_r"], params["lora_dropout"], params["lora_alpha"],
                     params["learning_rate"], params["warm_up_ratio"], params["weight_decay"])
    except Exception:
        loss = MAX_LOSS_VALUE
    return loss


def batch_iterator(dataset, batch_size):
    """
    Yields batches from a dataset.

    Args:
        dataset (Iterable): The dataset to split into batches.
        batch_size (int): The size of one batch.

    Yields:
        list: One batch of items from the dataset.
    """
    batch = []
    for item in dataset:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_chunked_dataset(dataset_id):
    """
    Loads and concatenates chunked train/test datasets from disk for larger datasets.

    Args:
        dataset_id (str): Name of the currently processed dataset.

    Returns:
        DatasetDict: A DatasetDict divided into train and test sets.

    Raises:
        FileNotFoundError: If any expected dataset chunk files are missing.
    """    
    try:
        # find paths to chunks of train set
        train_paths = sorted([
            f"{DATASET_DIRECTORY}/{dataset_id}/{d}" 
            for d in os.listdir(f"{DATASET_DIRECTORY}/{dataset_id}")
            if d.startswith("train_chunk_")
        ])

        # find paths to chunks of test set
        test_paths = sorted([
            f"{DATASET_DIRECTORY}/{dataset_id}/{d}"
            for d in os.listdir(f"{DATASET_DIRECTORY}/{dataset_id}")
            if d.startswith("test_chunk_")
        ])

        # load and concatenate train datasets
        train_datasets = [datasets.load_from_disk(path) for path in train_paths]
        train_dataset = datasets.concatenate_datasets(train_datasets)

        # load and concatenate test datasets
        test_datasets = [datasets.load_from_disk(path) for path in test_paths]
        test_dataset = datasets.concatenate_datasets(test_datasets)

        # create DatasetDict with train and test sets
        tokenized_dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
    except FileNotFoundError:
        raise FileNotFoundError(f"Some dataset chunk files are missing in {DATASET_DIRECTORY}/{dataset_id}")
        
    return tokenized_dataset


def load_slovak_dataset(tokenizer, dataset_id):
    """
    Loads and tokenizes a Slovak dataset for LLM fine-tuning. Handles loading from disk if available, or processes and
    tokenizes from JSONL if not. Supports streaming and chunked saving for large datasets.

    Args:
        tokenizer: Llama 3.1 tokenizer for tokenization.
        dataset_id (str): Name of the current dataset.

    Returns:
        DatasetDict: A DatasetDict with tokenized dataset divided into train and test sets.

    Raises:
        ValueError: If dataset name is not defined or is not a string.
        OSError: If directory with datasets is not accessible or does not exist.
        TypeError: If function for tokenization is not callable.
        FileNotFoundError: If the dataset file or dataset chunk folder is missing or is empty.
    """
    
    def tokenize_function(sentences):
        """
        Tokenizes input examples and prepares shifted labels for causal language modeling.
    
        Args:
            sentences (dict): A batch of sentences containing the "content" field.
    
        Returns:
            dict: Tokenized sentences with 'input_ids', 'attention_mask', and shifted 'labels'.
        """
        tokens = tokenizer(sentences["content"], padding="max_length", truncation=True)
        labels = tokens["input_ids"].copy()
        for k, label in enumerate(labels):
            labels[k] = label[1:] + [-100]
        tokens["labels"] = labels
        return tokens

    # validate that dataset_id is a non-empty string
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("The 'dataset_id' must be a non-empty string.")

    # validate that dataset directory is defined and points to a valid directory
    if not os.path.exists(DATASET_DIRECTORY):
        raise OSError(f"The directory {DATASET_DIRECTORY} does not exist or is not accessible.")

    # validate that tokenization function is callable
    if not callable(tokenize_function):
        raise TypeError("The 'tokenize_function' is not callable.")
        
    tokenized_dataset = None
    # load tokenized dataset from disk if available 
    try:
        # handling of larger datasets that are saved in form of chunks
        if dataset_id == "OPUS":
            tokenized_dataset = load_chunked_dataset(dataset_id)

        # loading of datasets that are saved in one file
        else:
            tokenized_dataset = datasets.load_from_disk(DATASET_DIRECTORY + "/" + dataset_id)
            
    except FileNotFoundError:
        # path to the dataset
        dataset_name = "datasets/" + dataset_id + ".jsonl"

        # tokenization of larger datasets
        if dataset_id == "OPUS" or dataset_id == "C4":
            # loading the dataset
            dataset = datasets.load_dataset("json", data_files=dataset_name, streaming=True)

            # iterating through dataset in batches of 100 000 items
            for j, batch in enumerate(batch_iterator(dataset["train"], batch_size=100000)):
                # 20% of the dataset is stored in test chunks, the rest is train dataset
                if j % 5 == 0:
                    save_path = DATASET_DIRECTORY + "/" + dataset_id + "/test_chunk_" + str(j)
                else:
                    save_path = DATASET_DIRECTORY + "/" + dataset_id + "/train_chunk_" + str(j)

                # tokenization of the batch
                dataset_batch = Dataset.from_list(batch)
                tokenized_batch = dataset_batch.map(tokenize_function, batched=True)
                # saving the tokenized batch on the disk
                tokenized_batch.set_format(type='torch',
                                           columns=['input_ids', 'attention_mask', 'labels'])
                tokenized_batch.save_to_disk(save_path)

            try:
                tokenized_dataset = load_chunked_dataset(dataset_id)
            except FileNotFoundError:
                raise FileNotFoundError(f"Some dataset chunk files are missing in {DATASET_DIRECTORY}/{dataset_id}")

        else:
            try:
                # loading the dataset and splitting it to train and test part with ratio 80:20
                dataset = datasets.load_dataset("json", data_files=dataset_name)
                dataset = dataset["train"].train_test_split(test_size=TRAIN_TEST_RATIO)

                # tokenization of the dataset
                tokenized_dataset = dataset.map(tokenize_function, batched=True)
                # tokenized_dataset = tokenized_dataset.flatten_indices()
                # saving the tokenized batch on the disk
                tokenized_dataset.set_format(type='torch', columns=['input_ids', 'labels'])
                tokenized_dataset.save_to_disk(DATASET_DIRECTORY + "/" + dataset_id)
            except Exception as e:
                print("Full error:", e)
        
    return tokenized_dataset
    

def load_model(model_id):
    """
    Loads a quantized Llama 3.1 model in 4-bit precision and prepares it for training.

    Args:
        model_id (str): Path to the pretrained model of Llama 3.1 8B.

    Returns:
        AutoModelForCausalLM: Prepared and quantized model ready for fine-tuning.
    Raises:
        ValueError: If `model_id` is invalid or empty.
        FileNotFoundError: If the model defined by `model_id` cannot be found or loaded.
    """
    # validate that model_id is a non-empty string
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("The 'model_id' must be a non-empty string.")

    try:
        # configuration of BitsAndBytes 4 bit quantization
        quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                          bnb_4bit_compute_dtype=torch.bfloat16,
                                          bnb_4bit_quant_type="nf4",
                                          bnb_4bit_use_double_quant=True)
    
        # loading of the pretrained base Llama 3.1 8B model in bfloat16 precision with quantization
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     use_cache=False,
                                                     low_cpu_mem_usage=True,
                                                     torch_dtype=torch.bfloat16,
                                                     attn_implementation=ATTENTION_MECHANISM,
                                                     quantization_config=quant_config)
    
        # preparing the model for quantized training and enabling gradient checkpointing
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"The model at {model_id} could not be found or loaded.")
    
    return model
    

def initialize_adapter(model, lora_r, lora_dropout, lora_alpha):
    """
    Initializes LoRA adapter configuration based on the currently set hyperparameters. The LoRA is applied to projection
    layers related to attention mechanism and feed-forward networks.

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
    # validate model type
    if not isinstance(model, LlamaForCausalLM):
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


def custom_set_seed(seed):
    """
    Sets the random seed for reproducibility across multiple GPUs.

    Args:
        seed (int): The seed value to initialize the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync_seeds(seed):
    """
    Synchronizes the random seed across all distributed processes.

    Args:
        seed (int): The seed value to be synchronized across devices.
    """
    if torch.distributed.is_initialized():
        seed_tensor = torch.tensor(seed, dtype=torch.int32, device="cuda")
        torch.distributed.broadcast(seed_tensor, src=0)
        seed = seed_tensor.item()
    custom_set_seed(seed)


def load_hyperparameters(model_checkpoint_path):
    """
    When continuing with the training, loads the hyperparameters from a JSON file stored in the model checkpoint.

    Args:
        model_checkpoint_path (str): Path to the model checkpoint directory.

    Returns:
        tuple: A tuple containing the hyperparameters:
            - lora_r (int)
            - lora_dropout (float)
            - lora_alpha (float)
            - learning_rate (float)
            - warm_up_ratio (float)
            - weight_decay (float)
    Raises:
        FileNotFoundError: If the JSON file or the checkpoint directory does not exist.
    """
    # validate that the checkpoint path exists
    if not os.path.isdir(model_checkpoint_path):
        raise FileNotFoundError(f"The directory {model_checkpoint_path} does not exist.")
    
    # validate that the hyperparameters json file exists
    hyperparameters_file = f"{model_checkpoint_path}/hyperparameters.json"
    if not os.path.isfile(hyperparameters_file):
        raise FileNotFoundError(f"The file {hyperparameters_file} does not exist.")
    
    # load the hyperparameters from the json file
    with open(hyperparameters_file, "r") as file:
        loaded_hyperparameters = json.load(file)

    # get the hyperparameter values from the dictionary
    lora_r = loaded_hyperparameters["lora_r"]
    lora_dropout = loaded_hyperparameters["lora_dropout"]
    lora_alpha = loaded_hyperparameters["lora_alpha"]
    learning_rate = loaded_hyperparameters["learning_rate"]
    warm_up_ratio = loaded_hyperparameters["warm_up_ratio"]
    weight_decay = loaded_hyperparameters["weight_decay"]

    return lora_r, lora_dropout, lora_alpha, learning_rate, warm_up_ratio, weight_decay


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


def load_model_checkpoint(model_checkpoint_path, model_path_rank):
    """
    Loads the model checkpoint and processes the state dictionary to match the state dictionary of the base model.

    Args:
        model_checkpoint_path (str): Path to the model checkpoint directory.
        model_path_rank (str): The specific model checkpoint file to load based on the GPU rank.

    Returns:
        dict: A dictionary containing the model's state dictionary.

    Raises:
        FileNotFoundError: If the model checkpoint file does not exist.
        KeyError: If the expected "module" key is not found in the checkpoint.
        ValueError: If the checkpoint is not in the expected format.
    """
    # check if the model checkpoint path is valid
    if not os.path.isdir(model_checkpoint_path):
        raise FileNotFoundError(f"The directory {model_checkpoint_path} does not exist.")
    
    # check if the model checkpoint file exists
    checkpoint_file = f"{model_checkpoint_path}/{model_path_rank}"
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"The file {checkpoint_file} does not exist.")

    # load the model checkpoint to CPU
    try:
        model_checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load the model checkpoint: {e}")

    # check if the checkpoint contains the expected "module" key
    if "module" not in model_checkpoint:
        raise KeyError(f"Checkpoint does not contain the expected 'module' key.")

    # adjust the state dictionary to match the base model by removing unnecessary prefixes
    state_dict = {}
    for key, value in model_checkpoint["module"].items():
        new_key = key.replace("base_model.model.", "")
        state_dict[new_key] = value

    return state_dict


def save_model_checkpoint(model_engine, output_id, lora_r, lora_dropout, lora_alpha, learning_rate, warm_up_ratio,
                          weight_decay):
    """
    Saves the model checkpoint along with hyperparameters to the specified output directory.

    Args:
        model_engine (DeepSpeedEngine): The DeepSpeed model engine used for training.
        output_id (str): Identifier for the checkpoint output directory in format of the datetime.
        lora_r (int): The rank parameter for LoRA.
        lora_dropout (float): The dropout rate for LoRA.
        lora_alpha (int): The scaling factor for LoRA.
        learning_rate (float): The learning rate used for training.
        warm_up_ratio (float): The warm-up ratio used for learning rate scheduling.
        weight_decay (float): The weight decay value used for optimization.

    Raises:
        ValueError: If any of the hyperparameters are of the incorrect type or value.
        FileNotFoundError: If the output directory does not exist.
        IOError: If saving of the checkpoint or hyperparameters fails.
    """
    # validate the existence of the output directory
    if not os.path.exists(MODEL_OUTPUT_DIRECTORY):
        raise FileNotFoundError(f"The directory {MODEL_OUTPUT_DIRECTORY} does not exist.")

    # validate output_id format (should be a non-empty string)
    if not isinstance(output_id, str) or not output_id.strip():
        raise ValueError("output_id must be a non-empty string.")

    # specify the path to the checkpoint and save the fine-tuned model
    model_output_path = MODEL_OUTPUT_DIRECTORY + "/" + output_id
    
    # Save the model checkpoint
    try:
        model_engine.save_checkpoint(model_output_path)
    except Exception as e:
        raise IOError(f"Failed to save the model checkpoint: {e}")

    # to save the configuration of the hyperparameters only once, check GPU rank
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        # create dictionary of hyperparameters
        hyperparameters = {"lora_r": int(lora_r), 
                           "lora_dropout": float(lora_dropout),
                           "lora_alpha": int(lora_alpha),
                           "learning_rate": float(learning_rate),
                           "warm_up_ratio": float(warm_up_ratio),
                           "weight_decay": float(weight_decay)}

        try:
            # Save hyperparameters in json format in the same directory as the model checkpoint
            with open(f"{model_output_path}/hyperparameters.json", "w") as file:
                json.dump(hyperparameters, file, indent=4)
        except Exception as e:
            raise IOError(f"Failed to save hyperparameters: {e}")
    

def train(lora_r, lora_dropout, lora_alpha, learning_rate, warm_up_ratio, weight_decay, model_checkpoint_path=None):
    """
    Fine-tunes an LLM with LoRA adapters on the Slovak language dataset, utilizing DeepSpeed for distributed training.

    Args:
        lora_r (int): The rank parameter for LoRA.
        lora_dropout (float): The dropout rate for LoRA.
        lora_alpha (int): The scaling factor for LoRA.
        learning_rate (float): The learning rate for training.
        warm_up_ratio (float): The warm-up ratio for learning rate scheduling.
        weight_decay (float): The weight decay for the optimizer.
        model_checkpoint_path (str, optional): Path to the model checkpoint, if continuing the training.

    Returns:
        float: The average loss of the validation set after training as the objective for hyperparameter optimization.

    Raises:
        Exception: If any problem occurred during the training loop.
    """
    sync_seeds(42)
    # the checkpoints are identified based on the datetime corresponding to the start of training
    output_id = str(int(time.time()))
    # identification of the GPU on which the code is executed
    rank = int(os.environ.get("RANK", 0))

    # if continuing with training, load the hyperparameters
    if model_checkpoint_path:
        try:
            lora_r, lora_dropout, lora_alpha, learning_rate, warm_up_ratio, weight_decay = (
                load_hyperparameters(model_checkpoint_path))
        except FileNotFoundError:
            raise Exception(f"The checkpoint {model_checkpoint_path} was not saved together with the "
                            f"hyperparameter values")

    # specify paths for LoRA checkpoints
    lora_path = "adapters/" + output_id

    # initialize run in Wandb and log the current configuration of hyperparameters and other relevant values
    if rank == 0:
        wandb.init(
                project=os.getenv("WANDB_PROJECT_NAME"),
                entity=os.getenv("WANDB_USER_NAME"),
                config={"lora_r": lora_r,
                        "lora_dropout": lora_dropout,
                        "lora_alpha": lora_alpha,
                        "dataset": DATASET_NAME,
                        "learning_rate": learning_rate,
                        "warm_up_ratio": warm_up_ratio,
                        "epochs": NUMBER_OF_TRAINING_EPOCHS,
                        "optimizer": OPTIMIZER_NAME,
                        "model": MODEL_ID,
                        "weight_decay": weight_decay,
                        "output_id": output_id})

    # load the tokenizer of Llama 3.1 8B with enabled padding and truncation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                              padding=True,
                                              truncation=True,
                                              model_max_length=MAX_SEQUENCE_LENGTH,
                                              return_tensors="pt")
    # as not padding token is specified by the tokenized, it is set to EOS token
    tokenizer.pad_token = tokenizer.eos_token
    # load the base model in quantized format
    try:
        model = load_model(MODEL_ID)
    except (ValueError, FileNotFoundError):
        raise Exception(f"Not able to load model {MODEL_ID}.")

    # initialize the LoRA configuration and prepare the model of PEFT training
    try:
        lora_config = initialize_adapter(model, lora_r, lora_dropout, lora_alpha)
    except ValueError:
        raise Exception(f"Not able to initialize LoRA.")
    model = get_peft_model(model, lora_config)
    
    print(f"Memory footprint of the model: {model.get_memory_footprint() / 1024 ** 3} GB")
    model.print_trainable_parameters()

    # set the Adam optimizer and cosine learning rate scheduler
    optimizer = DeepSpeedCPUAdam(model.parameters(), 
                                 lr=learning_rate, 
                                 betas=(0.8, 0.999),
                                 weight_decay=weight_decay)
    lr_scheduler = get_scheduler("cosine",
                                 optimizer=optimizer,
                                 num_warmup_steps=NUMBER_OF_WARM_UP_STEPS,
                                 num_training_steps=NUMBER_OF_TRAINING_STEPS)

    # initialize training in distributed environment using DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        config=DEEPSPEED_CONFIG,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    # if continuing in training, load the state of the model
    if model_checkpoint_path:
        state_dict = None
        entries = os.listdir(model_checkpoint_path)
        global_step = [entry for entry in entries
                       if os.path.isdir(os.path.join(model_checkpoint_path, entry))]
        if len(global_step) != 1:
            raise Exception(f"The checkpoint path {model_checkpoint_path} does not contain global step.")

        # load the right state of the model based on the GPU the code is running on
        if rank == 0:
            try:
                state_dict = load_model_checkpoint(global_step[0], MODEL_PATH_RANK0)
            except (KeyError, ValueError, FileNotFoundError):
                raise Exception(f"Not able to load model from checkpoint {global_step[0]}/{MODEL_PATH_RANK0}")

        if rank == 1:
            try:
                state_dict = load_model_checkpoint(global_step[0], MODEL_PATH_RANK1)
            except (KeyError, ValueError, FileNotFoundError):
                raise Exception(f"Not able to load model from checkpoint {global_step[0]}/{MODEL_PATH_RANK1}")

        # assign the loaded state dictionary to the model in the distributed environment
        model_engine.module.load_state_dict(state_dict, strict=False)
        # set the LoRA parameters as trainable
        for name, param in model_engine.module.named_parameters():
            if 'lora' in name and param.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                param.requires_grad = True

    # Ensure model is spread across 2 GPUs
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_engine.to(device)

    # get the batch size per GPU from the DeepSpeed configuration
    batch_size = model_engine.train_micro_batch_size_per_gpu()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=batch_size)

    # load the Slovak dataset and tokenize it
    try:
        dataset = load_slovak_dataset(tokenizer, DATASET_NAME)
    except (ValueError, OSError, TypeError, FileNotFoundError):
        raise Exception(f"Not able to load {DATASET_NAME} dataset.")

    # initialize DataLoader for train and test sets of the tokenized datasets
    train_dataloader = DataLoader(
        dataset["train"], batch_size=batch_size, collate_fn=data_collator, num_workers=0
    )
    test_dataloader = DataLoader(
        dataset["test"], batch_size=batch_size, collate_fn=data_collator, num_workers=0
    )

    # calculate total number of training steps and warm-up steps and set those values in learning rate scheduler
    num_training_steps = (len(train_dataloader) // batch_size) * NUMBER_OF_TRAINING_EPOCHS
    num_of_warmup_steps = int(num_training_steps * warm_up_ratio)
    lr_scheduler.num_training_steps = num_training_steps
    lr_scheduler.num_warmup_steps = num_of_warmup_steps

    print(f"DeepSpeed active GPU count: {torch.cuda.device_count()}")
    try:
        patience = 50
        best_loss = np.inf

        # fine-tuning loop
        for epoch in range(NUMBER_OF_TRAINING_EPOCHS):
            total_loss = 0
            running_loss_window = deque(maxlen=patience)
            patience_counter = 0
            model_engine.train()
            print(train_dataloader)

            # iterate through the train set in batches
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                # move the batch to the corresponding GPU
                batch = {key: value.to(model_engine.device) for key, value in batch.items()}
                # replace potential NaN values in the batch with 0.0
                batch["input_ids"] = torch.nan_to_num(batch["input_ids"], nan=0.0)
                batch["labels"] = torch.nan_to_num(batch["labels"], nan=0.0)
                labels = batch["labels"].to(torch.long)

                # get output from the model and save the loss value
                outputs = model_engine(input_ids=batch["input_ids"], labels=labels)
                loss = outputs['loss']
                total_loss += loss.item()

                # perform backward propagation with step of the optimizer
                model_engine.backward(loss)
                model_engine.step()

                # adjust the learning rate
                lr_scheduler.step()
                if rank == 0:
                    wandb.log({"loss": loss.item(), "epoch": epoch, "step": step})

                # every 100 steps log the current GPU usage in MBs
                if step % 100 == 0:
                    memory_usage = torch.cuda.max_memory_allocated() / 1024 ** 2
                    if rank == 0:
                        wandb.log({"gpu_memory": memory_usage})
                    torch.cuda.empty_cache()

                if step > 3000:
                    # calculate the average loss value for the last 50 steps
                    running_loss_window.append(loss.item())
                    avg_running_loss = np.mean(running_loss_window)
    
                    # if average loss is better than the best one so far, store the best loss and save the LoRA adapter
                    if avg_running_loss < best_loss:
                        best_loss = avg_running_loss
                        patience_counter = 0 
                        model_engine.module.save_pretrained(lora_path)
                    # if the average loss is not improving, increase the patience
                    else:
                        patience_counter += 1
    
                    # if the model is not improving for too long, trigger early stopping
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at batch {step+1} in epoch {epoch+1}")
                        break

            # calculate the average loss and perplexity
            avg_loss = total_loss / len(train_dataloader)
            perplexity = torch.exp(torch.tensor(avg_loss))

            if rank == 0:
                wandb.log({"average_loss": avg_loss, "perplexity": perplexity})

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("OOM error")

    # save the checkpoint of the fine-tuned model and free the memory
    try:
        save_model_checkpoint(model_engine, output_id, lora_r, lora_dropout, lora_alpha, learning_rate, warm_up_ratio,
                              weight_decay)
    except (ValueError, IOError, FileNotFoundError):
        raise Exception(f"Not able to save the model checkpoint identified with {output_id}")
    torch.cuda.empty_cache()

    # evaluation of the model
    model_engine.eval()
    total_loss = 0

    with torch.no_grad():
        # iterate through the test set in batches
        for step, batch in enumerate(tqdm(test_dataloader)):
            # move the batch to the corresponding GPU
            batch = {key: value.to(model_engine.device) for key, value in batch.items()}
            labels = batch["labels"].to(torch.long)

            # get output from the model and save the loss value
            outputs = model_engine(input_ids=batch["input_ids"], labels=labels)
            loss = outputs['loss']
            total_loss += loss.item()

            if step > len(test_dataloader) / 10:
                break

            if rank == 0:
                wandb.log({"valid_loss": loss})

    # calculate the average loss and perplexity
    avg_loss = total_loss / (len(test_dataloader) / 10)
    perplexity = torch.exp(torch.tensor(avg_loss))
    if rank == 0:
        wandb.log({"average_valid_loss": avg_loss, "valid_perplexity": perplexity})
        wandb.finish()

    # delete all the artefacts of the training and free memory
    del model, tokenizer, train_dataloader, test_dataloader, dataset, model_engine
    flush()
    # evaluate the fine-tuned model
    evaluate(MODEL_OUTPUT_DIRECTORY + "/" + output_id, lora_path)
    flush()

    return avg_loss
    

def evaluate(model_output_path, lora_output_path):
    """
    Evaluates the fine-tuned model by generating text using a sample input.

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
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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
            
        model = get_peft_model(base_model, lora_config)
        peft_model = PeftModel.from_pretrained(model, lora_output_path)

        # evaluation example
        input_text = "Prezidentom Slovenskej republiky je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=50)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        # delete all the artefacts of the evaluation and free memory
        del tokenizer, base_model, model, peft_model
        flush()
    

if __name__ == "__main__":
    load_dotenv()
    # log into the Hugging Face platform
    login(token=os.getenv("HUGGING_FACE_TOKEN"))
    # disable parallelism for tokenizers to avoid potential issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    flush()

    # run hyperparameter optimization
    results = skopt.forest_minimize(objective, space, n_calls=NUMBER_OF_HPO_CALLS, random_state=42)
    print(f"Best hyperparameters: {results.x}")
    
    for i in range(len(results.x_iters)):
        print(f"Run {i+1}: Params = {results.x_iters[i]}, Score = {results.func_vals[i]}")

    # save the results
    skopt.dump(results, HPO_DIRECTORY + "/results.pkl")

    # visualizations of the best hyperparameters' combinations
    skopt.plots.plot_convergence(results)
    skopt.plots.plot_evaluations(results)
    skopt.plots.plot_objective(results)
    