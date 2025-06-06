import pandas as pd
import numpy as np

# 시드 고정
np.random.seed(42)

n_students = 1000

# 1. 각 학생의 기저 수학 능력을 생성 (평균 75, 표준편차 10; 너무 낮은 점수를 피하기 위해 최소 40으로 클리핑)
base_ability = np.clip(np.random.normal(loc=75, scale=10, size=n_students), 40, 100)

# 2. 시험별 노이즈 추가
def generate_test_scores(base, noise_std):
    # 각 학생의 시험 점수 = 기저 능력 + 시험별 노이즈
    scores = base + np.random.normal(loc=0, scale=noise_std, size=base.shape)
    # 0~100 범위로 클리핑 후 정수 변환
    return np.clip(np.round(scores), 0, 100).astype(int)

# 3. 각 시험별 점수 생성
score_6월 = generate_test_scores(base_ability, noise_std=7)
score_9월 = generate_test_scores(base_ability, noise_std=5)
score_10월 = generate_test_scores(base_ability, noise_std=5)
score_수능 = generate_test_scores(base_ability, noise_std=3)

# 4. 데이터프레임 생성 (학생 ID와 각 시험 점수)
df_math = pd.DataFrame({
    'student_id': [f"S{1000+i}" for i in range(n_students)],
    '6월_수학': score_6월,
    '9월_수학': score_9월,
    '10월_수학': score_10월,
    '수능_수학': score_수능
})

# 결과 출력
import ace_tools as tools
tools.display_dataframe_to_user(name="수학 성적 예측용 데이터셋", dataframe=df_math)
