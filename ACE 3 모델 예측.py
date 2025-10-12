import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# --- 1. 설정 및 모델 클래스 정의 (변경 없음) ---
@dataclass
class CFG:
    model_name: str = "klue/roberta-base"
    max_length: int = 128
    dropout: float = 0.1
    tokenizer_extra_special_tokens = ["[ENT]", "[/ENT]"]
    emotion_num_labels: int = 3
    entity_num_labels: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./ace_model_1"

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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits_emotion = self.emotion_classifier(pooled_output)
        logits_toxicity = self.toxicity_classifier(pooled_output).squeeze(-1)
        logits_entity = self.entity_classifier(pooled_output)
        return {"logits_emotion": logits_emotion, "logits_toxicity": logits_toxicity, "logits_entity": logits_entity}

# --- 2. 예측을 위한 준비 (변경 없음) ---
candidate_dict = {
    "이재명": [
        "이재명", "리짜이밍", "리짜이명", "찢", "전과 4범", "전과4범", 
        "재매이", "째매이", "형보수지", "이극단", "짜이밍햄", "대깨명", 
        "재매이햄", "재메이햄", "찢재명", "찢재매이", "찢재", "극단제명", 
        "찢자이밍", "재며이", "재맹이", "죄명이", "리재명", "이재멍", 
        "이럼통", "이재맹", "리좨맹", "쟤밍이", "찢죄명", "이죄명", 
        "쩨이메이", "리쟈오밍", "리쒜쒜",
        "킹재명", "갓재명"
    ],
    "이준석": [
        "이준석", "펨코남신", "펨코 남신", "준쎽", "준스톤", "준슥이햄", 
        "즌석", "대준석", "준서기", "준슥이형", "준섯", "준슥", "개준석", 
        "준스기", "준 스기", "준식이", "이준돌", "이준삭", "준씨기",
        "킹스톤", "빛준석", "돌준석"
    ],
    "김문수": [
        "김문수", "문순대", "김푼수", "김문석", "김먼수", "김문순대"
    ],
    "권영국": [
        "권영국", "권후보", "권망상", "권씨"
    ],
}
idx2emotion = {0: "긍정", 1: "부정", 2: "중립"}
idx2toxicity = {0: "정상", 1: "욕설"}
entity_list = list(candidate_dict.keys())
entity_list.append("불명")
idx2entity = {i: name for i, name in enumerate(entity_list)}

# --- 3. ★★★★★★★★★★★★ 수정된 예측 로직 ★★★★★★★★★★★★ ---

def find_and_mark_entity_for_prediction(comment, candidate_dict):
    """댓글에서 후보자 별칭을 찾아 마킹하고, 찾은 후보자 이름을 반환하는 함수"""
    # 모든 후보자의 모든 별칭을 하나의 리스트로 만듦 (긴 것부터 정렬)
    all_aliases = []
    for name, aliases in candidate_dict.items():
        for alias in aliases:
            all_aliases.append((alias, name))
    all_aliases.sort(key=lambda x: len(x[0]), reverse=True)

    # 댓글에서 별칭 검색
    for alias, name in all_aliases:
        pattern = re.compile(re.escape(alias))
        if pattern.search(comment):
            # 첫 번째 발견된 별칭을 마킹하고, (마킹된 텍스트, 후보자 이름) 반환
            marked_comment = pattern.sub(r"[ENT] \g<0> [/ENT]", comment, count=1)
            return marked_comment, name
    
    # 어떤 후보도 찾지 못하면 (원본 텍스트, '불명') 반환
    return comment, "불명"

def predict_from_csv(cfg, model, tokenizer, df):
    model.to(cfg.device)
    model.eval()

    # 1. 댓글을 순회하며 후보자 탐지 및 엔티티 마킹 수행
    processed_data = [find_and_mark_entity_for_prediction(str(comment), candidate_dict) for comment in df['comment']]
    
    # 마킹된 텍스트와, 탐지된 후보자 이름을 분리
    marked_comments = [item[0] for item in processed_data]
    found_entities = [item[1] for item in processed_data]

    # 2. 모델 예측 수행
    emotion_results, toxicity_results, entity_model_results = [], [], []
    batch_size = 32
    for i in tqdm(range(0, len(marked_comments), batch_size), desc="Predicting"):
        batch_comments = marked_comments[i:i+batch_size]
        
        inputs = tokenizer(
            batch_comments,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
        ).to(cfg.device)

        with torch.no_grad():
            outputs = model(**inputs)

        emotion_preds = torch.argmax(outputs['logits_emotion'], dim=-1).cpu().numpy()
        toxicity_preds = (torch.sigmoid(outputs['logits_toxicity']) > 0.5).cpu().numpy()
        entity_preds = torch.argmax(outputs['logits_entity'], dim=-1).cpu().numpy()

        emotion_results.extend([idx2emotion[p] for p in emotion_preds])
        toxicity_results.extend([idx2toxicity[p] for p in toxicity_preds])
        entity_model_results.extend([idx2entity[p] for p in entity_preds])

    # 3. 원본 데이터와 모든 결과 통합
    result_df = pd.DataFrame({
        'comment': df['comment'], # 원본 댓글 포함
        'found_entity_before_model': found_entities, # 1단계에서 찾은 후보자
        'predicted_emotion': emotion_results,
        'predicted_toxicity': toxicity_results,
        'predicted_entity_by_model': entity_model_results, # 모델이 최종 예측한 후보자
    })
    
    return result_df

# --- 4. 스크립트 실행 (★★★★★ 이 부분으로 교체 ★★★★★) ---
if __name__ == "__main__":
    # 입출력 파일 이름 정의
    input_filename = "input.csv"
    output_filename = "output_ACE.csv"
    cfg = CFG()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": cfg.tokenizer_extra_special_tokens})
    
    model = MultiTaskModel(cfg)
    model.encoder.resize_token_embeddings(len(tokenizer))
    
    model_path = f"{cfg.output_dir}/best_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        print(f"'{model_path}' 에서 학습된 모델을 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print(f"오류: '{model_path}' 파일을 찾을 수 없습니다. 모델 학습을 먼저 완료해주세요.")
        exit()

    try:
        # ★★★ 수정된 부분: encoding='utf-8-sig' 추가 ★★★
        # CSV 파일을 읽을 때 인코딩 방식을 명시해줍니다.
        input_df = pd.read_csv(input_filename, encoding='utf-8-sig')
        print(f"'{input_filename}' 파일을 성공적으로 불러와 예측을 시작합니다.")
        
        # 만약 위 코드로도 comment가 비어있다면, 아래 옵션으로 시도해보세요.
        # input_df = pd.read_csv(input_filename, encoding='utf-8')
        # input_df = pd.read_csv(input_filename, encoding='cp949')

    except FileNotFoundError:
        print(f"오류: '{input_filename}' 파일을 찾을 수 없습니다. 'comment' 열이 있는 CSV 파일을 준비해주세요.")
        exit()
    except Exception as e:
        print(f"파일을 읽는 중 다른 오류가 발생했습니다: {e}")
        exit()
        
    result_df = predict_from_csv(cfg, model, tokenizer, input_df)
    result_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"예측 완료! 결과가 '{output_filename}' 파일에 저장되었습니다.")