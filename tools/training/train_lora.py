import os
# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# IMPORTANT: unsloth must be imported before transformers so it can apply
# its memory/speed patches. Importing transformers first (e.g. to monkeypatch
# it) silently disables those patches and causes real OOMs.
from unsloth import FastLanguageModel

# Work around a transformers 5.x bug where caching_allocator_warmup
# over-allocates memory before 4-bit quantization is applied, causing
# spurious CUDA OOM errors on small GPUs even though the quantized
# model would easily fit. This just disables the (purely perf-related)
# upfront warmup allocation; the model still loads correctly.
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *args, **kwargs: None
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

def train_amadeus():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(base_dir, 'data', 'kurisu_finetune.jsonl')
    output_dir = os.path.join(base_dir, 'models', 'amadeus-kurisu-lora')

    print("Loading base model (Qwen 2.5 7B)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", # pre-quantized checkpoint avoids the bf16->4bit transient memory spike during loading
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = True,
        device_map = {"": 0}, # bitsandbytes 4-bit can't split across CPU/GPU without fp32 offload, so pin to GPU
    )

    print("Injecting LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3435,
    )

    print("Formatting dataset...")
    def formatting_prompts_func(examples):
        texts = []
        for instruction, input_text, output_text in zip(examples["instruction"], examples["input"], examples["output"]):
            text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
            texts.append(text)
        return { "text" : texts, }

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    print("Starting training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 1024,
        dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 4, # Changed from max_steps to full epochs
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3435,
            output_dir = "outputs",
        ),
    )

    trainer.train()

    print(f"Saving Amadeus LoRA to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete! The brain is ready.")

if __name__ == "__main__":
    train_amadeus()