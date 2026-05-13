import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration
from tqdm import tqdm

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # The model was saved in the parent directory in this specific training run
    model_path = os.path.join(base_dir, "..", "t5_gondi_model")
    
    print("=" * 60)
    print(f" Loading T5 model and tokenizer from {model_path}...")
    print("=" * 60)
    
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    csv_path = os.path.join(base_dir, "..", "..", "Datasets", "gondi_training_dataset.csv")
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # NOTE: The T5 model was trained on Gondi -> English in the last run.
    # If it was trained on Hindi, change "English" to "Hindi".
    df = df.dropna(subset=["Gondi", "English"]) 
    
    # Sample 1000 random rows for evaluation
    df_sample = df.sample(n=1000, random_state=42)
    
    prefix = "translate Gondi to English: "
    
    true_tokens_all = []
    pred_tokens_all = []
    
    references = []
    hypotheses = []
    
    print("\nGenerating predictions for 1000 samples (This may take a minute)...")
    for idx, row in tqdm(enumerate(df_sample.itertuples()), total=1000):
        src_text = str(row.Gondi).strip()
        tgt_text = str(row.English).strip()
        
        if not src_text or not tgt_text:
            continue
            
        input_text = prefix + src_text
        inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
            
        pred_ids = outputs[0].tolist()
        
        # For BLEU score: We decode the IDs back to strings and split into words
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
        
        ref_words = tgt_text.split()
        pred_words = pred_text.split()
        references.append([ref_words])
        hypotheses.append(pred_words)
        
        # For Confusion Matrix: We need token IDs for subwords
        tgt_ids = tokenizer(tgt_text, max_length=64, truncation=True).input_ids
        
        # Remove special tokens (like PAD, EOS, BOS, UNK)
        special_tokens = [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id, getattr(tokenizer, 'unk_token_id', -1)]
        tgt_ids_clean = [tid for tid in tgt_ids if tid not in special_tokens and tid is not None]
        pred_ids_clean = [tid for tid in pred_ids if tid not in special_tokens and tid is not None]
        
        # Match lengths to compare true vs predicted tokens directly
        min_len = min(len(tgt_ids_clean), len(pred_ids_clean))
        true_tokens_all.extend(tgt_ids_clean[:min_len])
        pred_tokens_all.extend(pred_ids_clean[:min_len])

    # --- 1. Calculate BLEU Score ---
    try:
        smoothie = SmoothingFunction().method4
        bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothie) * 100
        print(f"\n" + "=" * 40)
        print(f" 🏆 BLEU Score (1000 samples): {bleu_score:.2f}")
        print("=" * 40 + "\n")
    except Exception as e:
        print(f"Error calculating BLEU: {e}")

    # --- 2. Calculate Confusion Matrix ---
    if not true_tokens_all:
        print("No valid tokens collected for confusion matrix!")
        return

    from collections import Counter
    true_counts = Counter(true_tokens_all)
    top_n = 20
    top_tokens = [tok for tok, count in true_counts.most_common(top_n)]
    
    other_token_id = -1
    
    def map_token(tid):
        return tid if tid in top_tokens else other_token_id

    mapped_true = [map_token(tid) for tid in true_tokens_all]
    mapped_pred = [map_token(tid) for tid in pred_tokens_all]
    
    labels = top_tokens + [other_token_id]
    label_names = []
    
    for tid in labels:
        if tid == other_token_id:
            label_names.append("OTHER")
        else:
            token_str = tokenizer.decode([tid])
            if not token_str.strip():
                token_str = f"ID:{tid}"
            label_names.append(token_str)
            
    cm = confusion_matrix(mapped_true, mapped_pred, labels=labels)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Tokens')
    plt.ylabel('True Tokens')
    plt.title(f'T5 Model Confusion Matrix for Top {top_n} Tokens (1000 samples)')
    plt.tight_layout()
    
    out_img = os.path.join(base_dir, 't5_confusion_matrix.png')
    plt.savefig(out_img, dpi=300)
    print(f"✅ Saved confusion matrix image as:\n   {out_img}")

if __name__ == "__main__":
    main()
