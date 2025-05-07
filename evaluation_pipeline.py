import torch
import gc
import os
import json
import numpy as np

from transformers import (AutoTokenizer,
                          AutoModelForCausalLM)
from peft import (LoraConfig,
                  get_peft_model,
                  PeftModel,
                  TaskType)

MODEL_ID = "meta-llama/Llama-3.1-8B"


def flush():
    """
    Frees up unused GPU memory and resets memory statistics. It helps OOM errors by explicitly collecting garbage to free Python memory,
    clearing the CUDA cache, collecting inter-process resources, and resetting peak memory usage statistics.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.ipc_collect()


def initialize_adapter(lora_r, lora_dropout, lora_alpha):
    """
    Initializes LoRA adapter configuration based on the currently set hyperparameters. The LoRA is applied to projection layers
    related to attention mechanism and feed-forward networks.

    Args:
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


def evaluate(lora_output_path):
    """
    Evaluates the fine-tuned model by generating text using a sample input.

    Args:
        lora_output_path (str): Path to the saved LoRA adapter.
    Raises:
        ValueError: If the model or LoRA checkpoint paths are invalid or missing.
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

        # Slovak language comprehension testing
        print("SLOVAK LANGUAGE COMPREHENSION")
        input_text = "Prezidentom Slovenskej republiky je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=100)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Dĺžka funkčného obdobia generálneho prokurátora na Slovensku je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=100)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Výška DPH na Slovensku je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=100)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Politické zriadenie Slovenskej republiky je klasifikované ako"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=100)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Dĺžka povinnej školskej dochádzky na Slovensku je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=100)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Slovensko je krajina, ktorá"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=100)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Medzi najznámejšie hrady na Slovenku patria"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=100)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Na Slovensku na oplatí navštíviť"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=100)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        # consistency testing
        print("CONSISTENCY TESTING")
        input_text = "Prezidentom Slovenskej republiky je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=50)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Prezidentom Slovenskej republiky je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=50)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Prezidentom Slovenskej republiky je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=50)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Prezidentom Slovenskej republiky je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=50)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Prezidentom Slovenskej republiky je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=50)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        print("CONTEXTUAL UNDERSTANDING")
        print("PERSON:")
        input_text = "Prezidentom Slovenskej republiky je Peter Pellegrini. Meno osoby v tejto vete je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=10)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Dekanom Fakulty informatiky a informačných technológií v Bratislave je prof. Ing. Ivan Kotuliak, PhD.. Meno osoby v tejto vete je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=10)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        print("LOCATION")
        input_text = "Najväčším mestom v mojom okolí je Viedeň. Názov lokality v tejto vete je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=10)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Sídlom Ministerstva zdravotníctva, na ktorého čele stojí od 10. októbra 2024 Kamil Šaško, Msc., je Bratislava. Názov lokality v tejto vete je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=10)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        print("ORGANIZATION")
        input_text = "Mensa Slovensko združuje nadpriemerne inteligentných ľudí. Názov organizácie v tejto vete je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=10)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Nadácia zastavme korupciu so sídlom v Bratislave bojuje proti korupcii už od roku 2014. Názov organizácie v tejto vete je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=10)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        print("INSTITUTION")
        input_text = "Najnovším ministerstvom je Ministerstvo cestovného ruchu a športu. Názov inštitúcie v tejto vete je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=20)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Národná rada Slovenskej republiky sídli v Bratislave a od 1. októbra 1992 je miestom zasadania poslancov. Názov inštitúcie v tejto vete je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=20)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        print("EVENT")
        input_text = "Voľby prezidenta sa uskutočnili v roku 2024. Názov udalosti v rámci tejto vety je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=20)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Súťaž tímových projektov TP Cup sa každoročne koná na Fakulte informatiky a informačných technológií. Názov udalosti v rámci tejto vety je"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=20)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        # comparison of the datasets
        input_text = "Slovensko je krajina, ktorá sa vyznačuje"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=100)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        input_text = "Medzi najznámejšie hrady na Slovenku patria"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output = peft_model.generate(input_ids, max_new_tokens=100)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        # delete all the artefacts of the evaluation and free memory
        del tokenizer, base_model, model, peft_model
        flush()


if __name__ == "__main__":
    login(token=os.getenv("HUGGING_FACE_TOKEN"))
    flush()
    # C4 dataset
    evaluate("adapters/C4")

    # OPUS dataset
    evaluate("adapters/OPUS")

    # BUTCorpus
    evaluate("adapters/BUTCorpus")

    # News dataset
    evaluate("adapters/News")

    # Gov dataset
    evaluate("adapters/Gov")

    # Books dataset
    evaluate("adapters/Books")

    # SlovLex dataset
    evaluate("adapters/SlovLex")
