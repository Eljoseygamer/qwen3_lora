import sys
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
from utils import load_input, write_predictions

BASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'base_model')
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), 'adapters')
BATCH_SIZE = 16
MAX_LEN = 512


def predict(texts, model, tokenizer, device):
    scores = []
    model.eval()
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc='Predicting'):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, truncation=True, max_length=MAX_LEN,
                padding=True, return_tensors='pt')
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
        scores.extend(probs.tolist())
    return scores


def main():
    input_dir = os.environ.get('inputDataset') or (sys.argv[1] if len(sys.argv) > 1 else '/tira-data/input')
    output_dir = os.environ.get('outputDir') or (sys.argv[2] if len(sys.argv) > 2 else '/tira-data/output')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    print(f'Loading Qwen3 base from {BASE_MODEL_DIR}...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_DIR, num_labels=2, quantization_config=bnb,
        device_map='auto', local_files_only=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f'Loading LoRA adapters from {ADAPTER_DIR}...')
    model = PeftModel.from_pretrained(model, ADAPTER_DIR, is_trainable=False, local_files_only=True)
    model.eval()

    print('Loading test data...')
    items = load_input(input_dir)
    ids = [x[0] for x in items]
    texts = [x[1] for x in items]

    scores = predict(texts, model, tokenizer, device)
    write_predictions(output_dir, ids, scores)
    print('Done.')


if __name__ == '__main__':
    main()
