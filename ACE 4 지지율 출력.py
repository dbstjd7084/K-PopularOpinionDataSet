import pandas as pd
import matplotlib.pyplot as plt
import platform
import os

# --- 설정 ---
INPUT_CSV = "output_ACE.csv"
CHART_FILENAME = "support_rate_chart.png"

# --- 한글 폰트 설정 함수 ---
def set_korean_font():
    """운영체제에 맞는 한글 폰트를 설정합니다."""
    system_name = platform.system()
    if system_name == "Windows":
        plt.rc("font", family="Malgun Gothic")
    elif system_name == "Darwin": # Mac OS
        plt.rc("font", family="AppleGothic")
    else: # Linux
        # 나눔 폰트가 설치되어 있는지 확인 후 설정
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if os.path.exists(font_path):
            plt.rc("font", family="NanumGothic")
        else:
            print("한글 폰트가 설치되어 있지 않아, 라벨이 깨질 수 있습니다.")
            print("Linux의 경우 'sudo apt-get install fonts-nanum*' 명령어로 폰트를 설치해주세요.")
    
    # 마이너스 부호 깨짐 방지
    plt.rc("axes", unicode_minus=False)

# --- 메인 분석 로직 ---
if __name__ == "__main__":
    # 한글 폰트 설정
    set_korean_font()

    # 1. 데이터 불러오기
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"오류: '{INPUT_CSV}' 파일을 찾을 수 없습니다.")
        print("모델 예측 스크립트를 먼저 실행하여 결과 파일을 생성해주세요.")
        exit()

    # '불명'을 제외한 후보자 목록 추출
    candidates = df[df['predicted_entity_by_model'] != '불명']['predicted_entity_by_model'].unique()

    results = []
    # 2. 후보자별 지지율 계산
    for candidate in candidates:
        # 해당 후보자에 대한 댓글만 필터링
        df_candidate = df[df['predicted_entity_by_model'] == candidate]
        
        # 긍정, 부정 댓글 수 계산
        positive_count = (df_candidate['predicted_emotion'] == '긍정').sum()
        negative_count = (df_candidate['predicted_emotion'] == '부정').sum()
        
        # 지지율 계산 (긍정 댓글 / (긍정 + 부정 댓글))
        total_polar_comments = positive_count + negative_count
        if total_polar_comments > 0:
            support_rate = (positive_count / total_polar_comments) * 100
        else:
            support_rate = 0  # 긍정/부정 댓글이 모두 없는 경우

        results.append({
            "후보자": candidate,
            "긍정 댓글 수": positive_count,
            "부정 댓글 수": negative_count,
            "지지율 (%)": support_rate
        })
    
    # 3. 결과 출력
    if not results:
        print("분석할 후보자 데이터가 없습니다.")
    else:
        df_results = pd.DataFrame(results).sort_values(by="지지율 (%)", ascending=False)
        print("--- 후보자별 댓글 지지율 분석 결과 ---")
        print(df_results.to_string(index=False))

        # 4. 결과 시각화 및 저장
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_results["후보자"], df_results["지지율 (%)"], color='skyblue')
        plt.title('후보자별 댓글 지지율 분석', fontsize=16)
        plt.xlabel('후보자', fontsize=12)
        plt.ylabel('지지율 (%)', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 막대 위에 값 표시
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

        plt.savefig(CHART_FILENAME)
        print(f"\n분석 결과 차트가 '{CHART_FILENAME}' 파일로 저장되었습니다.")
