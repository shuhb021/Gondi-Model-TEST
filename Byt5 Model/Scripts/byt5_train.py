import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "Datasets", "gondi_training_60k.csv")
    tokenizer_path = os.path.join(base_dir, "tokenizer.json")
    
    print("=" * 60)
    print(f" Loading BPE Tokenizer from {tokenizer_path}...")
    print("=" * 60)
    
    # Load the existing BPE tokenizer
    # We use PreTrainedTokenizerFast to load the tokenizer.json directly
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="</s>",
        bos_token="[START]"
    )
    
    # Verify tokenizer vocab size
    vocab_size = len(tokenizer)
    print(f"Tokenizer loaded. Vocabulary size: {vocab_size}")

    print("\n" + "=" * 60)
    print(f" Loading data from {csv_path}...")
    print("=" * 60)
    
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    
    # Drop rows missing any of the 3 languages
    df = df.dropna(subset=["Gondi", "Hindi", "English"]).reset_index(drop=True)
    
    # Create multi-language translation tasks
    pairs = [
        ("Gondi", "English", "translate Gondi to English: "),
        ("Gondi", "Hindi", "translate Gondi to Hindi: "),
        ("Hindi", "Gondi", "translate Hindi to Gondi: "),
        ("English", "Gondi", "translate English to Gondi: ")
    ]
    
    all_data = []
    for src, tgt, prefix in pairs:
        temp_df = pd.DataFrame({
            "source": prefix + df[src].astype(str) + " </s>",
            "target": df[tgt].astype(str) + " </s>"
        })
        all_data.append(temp_df)
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total sentences for multi-language training: {len(combined_df)}")
    
    # 1. Prepare Dataset
    dataset = Dataset.from_pandas(combined_df)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    max_length = 64
    
    def preprocess_function(examples):
        inputs = examples["source"]
        targets = examples["target"]
        
        model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
        labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True)
        
        # Replace pad_token_id with -100 so it's ignored in loss computation
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("\nTokenizing the multi-language dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["source", "target"])
    
    # 2. Initialize Model (Using ByT5 architecture parameters but with BPE vocab)
    print("\nInitializing ByT5 architecture with BPE vocabulary...")
    # ByT5-Small architecture: 12 layers encoder, 4 layers decoder (unlike T5 which is 6/6)
    # But here we can define a custom efficient config.
    config = T5Config(
        vocab_size=vocab_size,
        d_model=384,        # ByT5 small uses 1472, but for 60k we can use 384 or 512
        d_ff=1024,
        d_kv=64,
        num_layers=8,       # ByT5 uses more encoder layers
        num_decoder_layers=4,
        num_heads=6,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="gated-gelu", # ByT5 uses gated-gelu
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
    )
    
    model = T5ForConditionalGeneration(config)
    
    # 3. Training Arguments
    output_dir = os.path.join(base_dir, "byt5_gondi_results")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=5, # Set to 5 for a quick run, can be increased
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none"
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    print("\n" + "=" * 60)
    print(" Starting ByT5 (BPE) training loop...")
    print("=" * 60)
    
    # Auto-resume from the latest checkpoint if one exists in the output dir.
    # This means you can safely Ctrl+C at any time and just re-run the script
    # to continue from exactly where you left off.
    checkpoint_dir = output_dir
    resume_checkpoint = None
    if os.path.isdir(checkpoint_dir):
        checkpoints = [
            os.path.join(checkpoint_dir, d)
            for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            resume_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"  ↳ Resuming from checkpoint: {resume_checkpoint}")
        else:
            print("  ↳ No checkpoint found. Starting fresh.")
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # 4. Save Final Model
    save_path = os.path.join(base_dir, "byt5_gondi_model")
    print(f"\nSaving final model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("✅ Done! ByT5 model and BPE tokenizer saved.")

if __name__ == "__main__":
    main()
