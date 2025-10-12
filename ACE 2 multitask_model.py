import os
import random
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# ====================================================
# 설정 (Configuration)
# ====================================================
@dataclass
class CFG:
    model_name: str = "klue/roberta-base"
    max_length: int = 512
    dropout: float = 0.1
    tokenizer_extra_special_tokens = ["[ENT]", "[/ENT]"]
    num_train_epochs: int = 10
    train_batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    adam_eps: float = 1e-8
    emotion_num_labels: int = 3
    entity_num_labels: int = 5
    loss_weights: Dict[str, float] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    output_dir: str = "./ace_model_2"

cfg = CFG()
if cfg.loss_weights is None:
    cfg.loss_weights = {"emotion": 1.0, "toxicity": 1.0, "entity": 1.0}

# ====================================================
# 재현성을 위한 시드 고정
# ====================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)

# ====================================================
# 모델 정의 (MultiTaskModel)
# ====================================================
class MultiTaskModel(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.model_name)
        
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(cfg.dropout)

        self.emotion_classifier = nn.Linear(hidden_size, cfg.emotion_num_labels)
        self.toxicity_classifier = nn.Linear(hidden_size, 1)
        self.entity_classifier = nn.Linear(hidden_size, cfg.entity_num_labels)

    def forward(self, input_ids=None, attention_mask=None, emotion_labels=None, toxicity_labels=None, entity_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        logits_emotion = self.emotion_classifier(pooled_output)
        logits_toxicity = self.toxicity_classifier(pooled_output).squeeze(-1)
        logits_entity = self.entity_classifier(pooled_output)

        total_loss = 0
        losses = {}
        
        if emotion_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_emotion = loss_fct(logits_emotion, emotion_labels)
            losses['emotion'] = loss_emotion * self.cfg.loss_weights['emotion']
            total_loss += losses['emotion']

        if toxicity_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss_toxicity = loss_fct(logits_toxicity, toxicity_labels.float())
            losses['toxicity'] = loss_toxicity * self.cfg.loss_weights['toxicity']
            total_loss += losses['toxicity']

        if entity_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_entity = loss_fct(logits_entity, entity_labels)
            losses['entity'] = loss_entity * self.cfg.loss_weights['entity']
            total_loss += losses['entity']
            
        return {
            "loss": total_loss if total_loss != 0 else None,
            "logits_emotion": logits_emotion,
            "logits_toxicity": logits_toxicity,
            "logits_entity": logits_entity,
            **losses,
        }

# ====================================================
# 데이터 전처리 함수 (★★★★★ 최종 해결 버전 ★★★★★)
# ====================================================
def prepare_data(cfg: CFG, df: pd.DataFrame, tokenizer):
    """
    Pandas DataFrame을 HuggingFace Dataset으로 변환합니다.
    반복되는 .map() 오류를 해결하기 위해, Pandas로 직접 데이터를 처리한 후 Dataset을 생성합니다.
    """
    # 1. 텍스트 컬럼을 먼저 토크나이징합니다.
    # tokenizer는 텍스트 리스트를 입력받아 한 번에 효율적으로 처리할 수 있습니다.
    print("Tokenizing texts...")
    tokenized_outputs = tokenizer(
        df['input_text'].tolist(),
        truncation=True,
        max_length=cfg.max_length,
    )

    # 2. 라벨 컬럼을 처리합니다.
    # .apply와 lambda 함수를 사용해 각 값을 안전하게 변환하고 리스트로 만듭니다.
    print("Processing labels...")
    emotions = df['emotion'].apply(lambda x: int(x) if pd.notna(x) else -100).tolist()
    toxicities = df['toxicity'].apply(lambda x: float(x) if pd.notna(x) else -100).tolist()
    entities = df['entity'].apply(lambda x: int(x) if pd.notna(x) else -100).tolist()

    # 3. 토크나이징 결과와 라벨을 합쳐서 최종 Dataset 객체를 생성합니다.
    print("Creating HuggingFace Dataset...")
    final_dataset = Dataset.from_dict({
        'input_ids': tokenized_outputs['input_ids'],
        'attention_mask': tokenized_outputs['attention_mask'],
        'emotion': emotions,
        'toxicity': toxicities,
        'entity': entities,
    })

    return final_dataset

def compute_metrics(preds, labels):
    results = {}
    if 'emotion' in preds and 'emotion' in labels and len(labels['emotion']) > 0:
        em_preds = np.argmax(preds['emotion'], axis=1)
        em_labels = labels['emotion']
        results['emotion_macro_f1'] = f1_score(em_labels, em_preds, average='macro', zero_division=0)
        results['emotion_accuracy'] = accuracy_score(em_labels, em_preds)
    if 'toxicity' in preds and 'toxicity' in labels and len(labels['toxicity']) > 0:
        tx_preds = (torch.sigmoid(torch.tensor(preds['toxicity'])).numpy() >= 0.5).astype(int)
        tx_labels = labels['toxicity']
        results['toxicity_f1'] = f1_score(tx_labels, tx_preds, zero_division=0)
        results['toxicity_recall'] = recall_score(tx_labels, tx_preds, zero_division=0)
    if 'entity' in preds and 'entity' in labels and len(labels['entity']) > 0:
        en_preds = np.argmax(preds['entity'], axis=1)
        en_labels = labels['entity']
        results['entity_macro_f1'] = f1_score(en_labels, en_preds, average='macro', zero_division=0)
        results['entity_accuracy'] = accuracy_score(en_labels, en_preds)
    return results

def train(cfg: CFG, model, tokenizer, train_dataset, val_dataset):
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
        
    model.to(cfg.device)
    data_collator = DataCollatorWithPadding(tokenizer, padding="longest")
    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_loader) * cfg.num_train_epochs * cfg.warmup_ratio), num_training_steps=len(train_loader) * cfg.num_train_epochs)

    best_val_score = -1
    for epoch in range(cfg.num_train_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            emotion_labels = batch.pop("emotion", None)
            toxicity_labels = batch.pop("toxicity", None)
            entity_labels = batch.pop("entity", None)
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                emotion_labels=emotion_labels if emotion_labels is not None and (emotion_labels != -100).any() else None,
                toxicity_labels=toxicity_labels if toxicity_labels is not None and (toxicity_labels != -100).any() else None,
                entity_labels=entity_labels if entity_labels is not None and (entity_labels != -100).any() else None
            )

            loss = outputs["loss"]
            if loss is not None:
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{cfg.num_train_epochs}, Train Loss: {avg_train_loss:.4f}")
        
        val_metrics = evaluate(cfg, model, val_loader)
        print(f"Validation Metrics: {val_metrics}")
        
        val_score = val_metrics.get('emotion_macro_f1', 0)
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pth"))
            print(f"Saved best model with validation score: {best_val_score:.4f}")

def evaluate(cfg: CFG, model, dataloader):
    model.eval()
    all_preds = {'emotion': [], 'toxicity': [], 'entity': []}
    all_labels = {'emotion': [], 'toxicity': [], 'entity': []}

    with torch.no_grad():
        for batch in dataloader:
            batch_labels = {k: v.cpu().numpy() for k,v in batch.items() if k in ['emotion', 'toxicity', 'entity']}
            batch_inputs = {k: v.to(cfg.device) for k,v in batch.items() if k not in ['emotion', 'toxicity', 'entity']}
            
            outputs = model(**batch_inputs)
            
            for task in all_preds.keys():
                if task in batch_labels and (batch_labels[task] != -100).any():
                    all_preds[task].extend(outputs[f'logits_{task}'].detach().cpu().numpy())
                    all_labels[task].extend(batch_labels[task][batch_labels[task] != -100])
    
    valid_preds = {k: v for k, v in all_preds.items() if v}
    valid_labels = {k: v for k, v in all_labels.items() if v}
    
    return compute_metrics(valid_preds, valid_labels)

# ====================================================
# 메인 실행 블록
# ====================================================
if __name__ == "__main__":
    try:
        df = pd.read_csv("preprocessed_data.csv")
    except FileNotFoundError:
        print("오류: 'preprocessed_data.csv' 파일을 찾을 수 없습니다.")
        print("먼저 preprocess.py를 실행하여 데이터 전처리를 완료했는지 확인해주세요.")
        exit()
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": cfg.tokenizer_extra_special_tokens})
    
    model = MultiTaskModel(cfg)
    model.encoder.resize_token_embeddings(len(tokenizer))
    
    dataset = prepare_data(cfg, df, tokenizer)
    split = dataset.train_test_split(test_size=0.2, seed=cfg.seed)
    train_dataset = split['train']
    val_dataset = split['test']

    print(f"Using device: {cfg.device}")
    print(f"Starting training for {cfg.num_train_epochs} epochs...")
    train(cfg, model, tokenizer, train_dataset, val_dataset)