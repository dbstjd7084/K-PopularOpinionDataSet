import pandas as pd
import re
import numpy as np
from tqdm import tqdm

# (candidate_dict, map, mark_entity 함수 정의 부분은 기존과 동일)
# ... (생략) ...
candidate_dict = {
    "이재명": ["이재명", "재명", "리짜이밍", "리짜이명", "찢", "전과 4범", "전과4범", "재매이", "째매이", "형보수지", "이극단", "짜이밍햄", "대깨명", "재매이햄", "재메이햄", "찢재명", "찢재매이", "찢재", "극단제명", "찢자이밍", "재며이", "재맹이", "죄명이", "리재명", "이재멍", "이럼통", "이재맹", "리좨맹", "쟤밍이", "찢죄명", "이죄명", "쩨이메이", "리쟈오밍", "리쒜쒜", "킹재명", "갓재명"],
    "이준석": ["이준석", "준석", "펨코남신", "펨코 남신", "준쎽", "준스톤", "준슥이햄", "즌석", "대준석", "준서기", "준슥이형", "준섯", "준슥", "개준석", "준스기", "준 스기", "준식이", "이준돌", "이준삭", "준씨기", "킹스톤", "빛준석", "돌준석"],
    "김문수": ["김문수", "문수", "문순대", "김푼수", "김문석", "김먼수", "김문순대"],
    "권영국": ["권영국", "영국", "권후보", "권망상", "권씨", "권 저넘", "권 저놈"],
}
emotion_map = {"긍정": 0, "부정": 1, "중립": 2}
toxicity_map = {"정상": 0, "욕설": 1}
entity_list = list(candidate_dict.keys())
entity_list.append("불명")
entity_map = {name: i for i, name in enumerate(entity_list)}

def mark_entity(row):
    comment = str(row['comment'])
    entity_label = str(row['entity'])
    if entity_label not in candidate_dict:
        return comment
    aliases = candidate_dict[entity_label]
    aliases.sort(key=len, reverse=True)
    for alias in aliases:
        pattern = re.compile(re.escape(alias))
        if pattern.search(comment):
            return pattern.sub(r"[ENT] \g<0> [/ENT]", comment, count=1)
    return comment


# --- 메인 실행 로직 (★★★★★ 처리 순서 수정 ★★★★★) ---
if __name__ == "__main__":
    input_filename = "merged_final.csv" 
    output_filename = "preprocessed_data.csv"

    print(f"'{input_filename}' 파일을 불러옵니다...")
    try:
        df = pd.read_csv(input_filename, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"오류: '{input_filename}' 파일을 찾을 수 없습니다.")
        exit()

    print("전처리를 시작합니다...")

    # 1. (먼저) 여러 후보가 포함된 데이터 복제
    print("1/5: 여러 후보가 포함된 데이터 복제 중...")
    expanded_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        entities = str(row['entity']).strip()
        if ', ' in entities:
            individual_entities = [e.strip() for e in entities.split(', ')]
            for entity_name in individual_entities:
                new_row = row.copy()
                new_row['entity'] = entity_name
                expanded_rows.append(new_row)
        else:
            expanded_rows.append(row)
    df_expanded = pd.DataFrame(expanded_rows)
    print(f"데이터 복제 완료. (원본: {len(df)}행 -> 복제 후: {len(df_expanded)}행)")

    # 2. (그 다음) 'entity' 컬럼의 별명을 공식 이름으로 자동 변환
    print("2/5: 'entity' 컬럼의 별명을 공식 이름으로 자동 변환 중...")
    alias_to_official_map = {}
    for official_name, aliases in candidate_dict.items():
        for alias in aliases:
            alias_to_official_map[alias] = official_name
            
    df_expanded['entity'] = df_expanded['entity'].apply(lambda x: alias_to_official_map.get(str(x).strip(), str(x).strip()))

    # 3. 엔티티 마킹
    print("3/5: 엔티티 마킹 중...")
    df_expanded['input_text'] = df_expanded.apply(mark_entity, axis=1)

    # 4. 라벨을 숫자로 변환
    print("4/5: 라벨을 숫자로 변환 중...")
    df_expanded['emotion_label_num'] = df_expanded['emotion'].map(emotion_map)
    df_expanded['entity_label_num'] = df_expanded['entity'].map(entity_map)
    df_expanded['toxicity_label_num'] = df_expanded['toxicity'].map(toxicity_map)

    # 5. 최종 파일 저장
    print(f"5/5: 결과를 '{output_filename}' 파일로 저장 중...")
    final_df = pd.DataFrame({
        'input_text': df_expanded['input_text'],
        'emotion': df_expanded['emotion_label_num'],
        'toxicity': df_expanded['toxicity_label_num'],
        'entity': df_expanded['entity_label_num']
    })
    
    final_df.fillna(-100, inplace=True)
    
    final_df['emotion'] = final_df['emotion'].astype(int)
    final_df['entity'] = final_df['entity'].astype(int)
    final_df['toxicity'] = final_df['toxicity'].astype(int)

    final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print("\n전처리 완료!")
    print(f"'{output_filename}' 파일이 생성되었습니다.")