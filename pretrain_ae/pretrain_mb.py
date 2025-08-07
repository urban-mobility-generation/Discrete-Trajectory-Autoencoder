import os
import math
import logging
import sys
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    BertTokenizerFast,
    BartForConditionalGeneration,
    BartConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ========== environment setup ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["WANDB_DISABLED"] = "true"

# ========== logging setup ==========
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ========== data loading ==========
file_path = "./public_dataset/final_didi_train_segment_generation/train_data/final_segments_all_train_data.pkl"
trajs = pd.read_pickle(file_path)

def parse_sequence(seq):
    if isinstance(seq, str):
        return [int(x) for x in seq.strip().split()]
    elif isinstance(seq, list):
        return seq
    else:
        raise ValueError("unknown sequence type")

all_sequences = trajs['unique_id_seq'].apply(parse_sequence).tolist()

# ========== construct vocab ==========
vocab_file = "bart_vocab.txt"
if not os.path.exists(vocab_file):
    tokens = set()
    for seq in all_sequences:
        for token in seq:
            tokens.add(str(token))
    tokens = sorted(list(tokens), key=lambda x: int(x))
    special_tokens = ["<s>", "</s>", "<pad>", "<mask>", "<unk>"]
    vocab_list = special_tokens + tokens
    with open(vocab_file, "w", encoding="utf-8") as f:
        for token in vocab_list:
            f.write(token + "\n")
    logger.info(f"vocab size：{len(vocab_list)}")
else:
    logger.info("vocab exists")

# ========== initialize Tokenizer ==========
tokenizer = BertTokenizerFast(vocab_file=vocab_file, do_lower_case=False)
tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "unk_token": "<unk>"
})
logger.info(f"Tokenizer size: {len(tokenizer)}")

# ========== build Dataset ==========
class TrajectoryDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=128):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        text = " ".join(str(token) for token in seq)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: encoding[key].squeeze(0) for key in encoding}
        item["labels"] = item["input_ids"].clone()
        return item

max_seq_length = 128
dataset = TrajectoryDataset(all_sequences, tokenizer, max_length=max_seq_length)
train_size = int(0.9 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

# ========== Customize Collator:  Masking / Infilling ==========
class MaskedTrajectoryCollator:
    def __init__(self, tokenizer, mask_prob=0.3, span_mask=True, max_span=3):
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_prob = mask_prob
        self.span_mask = span_mask
        self.max_span = max_span

    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        corrupted_input = input_ids.clone()

        for i in range(input_ids.size(0)):
            corrupted_input[i] = self.corrupt_sequence(input_ids[i])

        attention_mask = (corrupted_input != self.pad_token_id).long()

        return {
            "input_ids": corrupted_input,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def corrupt_sequence(self, seq):
        seq = seq.tolist()
        valid_len = len([x for x in seq if x != self.pad_token_id])
        indices = list(range(valid_len))

        num_mask = int(self.mask_prob * valid_len)
        random.shuffle(indices)

        corrupted = seq[:]
        i = 0
        while num_mask > 0 and i < len(indices):
            idx = indices[i]
            span_len = random.randint(1, self.max_span) if self.span_mask else 1
            for j in range(span_len):
                if idx + j < valid_len and corrupted[idx + j] != self.tokenizer.pad_token_id:
                    corrupted[idx + j] = self.mask_token_id
                    num_mask -= 1
            i += 1

        return torch.tensor(corrupted + [self.pad_token_id] * (len(seq) - len(corrupted)))

data_collator = MaskedTrajectoryCollator(tokenizer, mask_prob=0.3, span_mask=True)

# ========== initialize BART model ==========
vocab_size = len(tokenizer)
config = BartConfig(
    vocab_size=vocab_size,
    d_model=256,
    encoder_layers=4,
    decoder_layers=2,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
    d_ff=512,
    max_position_embeddings=max_seq_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
model = BartForConditionalGeneration(config)
model.resize_token_embeddings(vocab_size)

# ========== start training ==========
training_args = Seq2SeqTrainingArguments(
    output_dir="./bart_pretrain_masked_4_2",
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    evaluation_strategy="steps",
    eval_steps=5000,
    save_steps=5000,
    logging_steps=2000,
    learning_rate=1e-5,
    weight_decay=0.01,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

logger.info("*** Starting Masked BART Pretraining ***")
trainer.train()
model.save_pretrained("./bart_pretrain_masked_4_2/final_model")

# ========== evaluation ==========
logger.info("*** Evaluation ***")
metrics = trainer.evaluate()
metrics["eval_samples"] = len(eval_dataset)
try:
    metrics["perplexity"] = math.exp(metrics["eval_loss"])
except OverflowError:
    metrics["perplexity"] = float("inf")
logger.info(f"Evaluation metrics: {metrics}")
