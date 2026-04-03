"""
Kaggle 설문 데이터 기반 연봉 분석 엔진
탭2에서 사용하는 경력별/기술개수별/직무별 연봉 비율 계산
"""
import pandas as pd
import numpy as np

SALARY_MID_MAP = {
    '0-999': 500, '1,000-1,999': 1500, '2,000-2,999': 2500,
    '3,000-3,999': 3500, '4,000-4,999': 4500, '5,000-7,499': 6250,
    '7,500-9,999': 8750, '10,000-14,999': 12500, '15,000-19,999': 17500,
    '20,000-24,999': 22500, '25,000-29,999': 27500, '30,000-39,999': 35000,
    '40,000-49,999': 45000, '50,000-59,999': 55000, '60,000-69,999': 65000,
    '70,000-79,999': 75000, '80,000-89,999': 85000, '90,000-99,999': 95000,
    '100,000-124,999': 112500, '125,000-149,999': 137500,
    '150,000-199,999': 175000, '200,000-249,999': 225000,
    '250,000-299,999': 275000, '300,000-499,999': 400000,
    '500,000-999,999': 750000,
}

LANG_COLS = ['Python', 'R', 'SQL', 'C', 'C#', 'C++', 'Java', 'Javascript', 'Bash', 'PHP', 'MATLAB', 'Julia', 'Go']
ML_COLS = ['Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'Fast.ai', 'Xgboost', 'LightGBM', 'CatBoost', 'Caret', 'Tidymodels', 'JAX', 'PyTorch Lightning', 'Huggingface']
ALGO_COLS = ['Linear/Logistic Regression', 'Decision Trees/Random Forests', 'Gradient Boosting (XGB, LGBM)', 'Bayesian Approaches', 'Evolutionary Approaches', 'Dense Neural Networks (MLP)', 'CNN', 'GAN', 'RNN', 'Transformer (BERT, GPT)', 'Autoencoder (DAE, VAE)', 'Graph Neural Networks']
ALL_TECH_COLS = LANG_COLS + ML_COLS + ALGO_COLS

EXP_ORDER = ['< 1 years', '1-3 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']
EXP_LABELS = ['1년 미만', '1-3년', '3-5년', '5-10년', '10-20년', '20년+']

JOB_LABELS = {
    'Data Analyst (Business, Marketing, Financial, Quantitative, etc)': 'Data Analyst',
    'Data Scientist': 'Data Scientist',
    'Data Engineer': 'Data Engineer',
    'Machine Learning/ MLops Engineer': 'ML Engineer',
}


def tech_group(n):
    if n <= 2:
        return '1-2개'
    elif n <= 5:
        return '3-5개'
    elif n <= 10:
        return '6-10개'
    else:
        return '11개+'


def tech_group_order():
    return ['1-2개', '3-5개', '6-10개', '11개+']


class KaggleAnalyzer:
    """Kaggle 설문 데이터 분석 엔진"""

    def __init__(self, csv_path="data/KAGGLE_DATA.csv"):
        df = pd.read_csv(csv_path)
        df['salary_mid'] = df['연봉 (USD)'].map(SALARY_MID_MAP)
        df = df[df['salary_mid'].notna()].copy()

        # 기술 개수 (0개인 사람은 비개발 관리자일 수 있어 1-2개부터 시작)
        df['tech_count'] = df[ALL_TECH_COLS].notna().sum(axis=1)
        df['tech_group'] = df['tech_count'].apply(tech_group)

        # 직무 라벨 축약
        df['job_short'] = df['직함'].map(JOB_LABELS)

        self.df = df
        self.n_total = len(df)

    def get_exp_growth_curve(self, job_filter=None):
        """경력별 이전 단계 대비 증가율 + 누적 비율 반환"""
        sub = self.df
        if job_filter:
            sub = sub[sub['job_short'] == job_filter]

        base_med = sub[sub['코딩 경력(년)'] == '< 1 years']['salary_mid'].median()
        if pd.isna(base_med) or base_med == 0:
            return []

        result = []
        prev_med = None
        for exp, label in zip(EXP_ORDER, EXP_LABELS):
            cell = sub[sub['코딩 경력(년)'] == exp]
            if len(cell) < 10:
                continue
            med = cell['salary_mid'].median()
            cumulative = med / base_med * 100
            step = ((med / prev_med - 1) * 100) if prev_med else 0
            result.append({
                'exp': label,
                'median': med,
                'cumulative_pct': cumulative,
                'step_pct': step,
                'n': len(cell),
            })
            prev_med = med

        return result

    def get_tech_count_premium(self, exp_filter=None):
        """기술 개수 구간별 프리미엄 (1-2개 대비)"""
        sub = self.df[self.df['tech_count'] >= 1]  # 0개 제외
        if exp_filter:
            sub = sub[sub['코딩 경력(년)'] == exp_filter]

        base = sub[sub['tech_group'] == '1-2개']['salary_mid'].median()
        if pd.isna(base) or base == 0:
            return []

        result = []
        for g in tech_group_order():
            cell = sub[sub['tech_group'] == g]
            if len(cell) < 10:
                continue
            med = cell['salary_mid'].median()
            premium = (med / base - 1) * 100
            result.append({
                'group': g,
                'median': med,
                'premium_pct': premium,
                'n': len(cell),
            })
        return result

    def get_job_comparison(self):
        """직무별 중앙값 비교"""
        result = []
        for job_short in ['Data Analyst', 'Data Scientist', 'Data Engineer', 'ML Engineer']:
            sub = self.df[self.df['job_short'] == job_short]
            if len(sub) < 30:
                continue
            result.append({
                'job': job_short,
                'median': sub['salary_mid'].median(),
                'mean': sub['salary_mid'].mean(),
                'n': len(sub),
            })
        return result

    def get_education_premium(self):
        """학력별 프리미엄"""
        edu_order = ["Bachelor's degree", "Master's degree", "Doctoral degree", "Professional doctorate"]
        edu_labels = ["학사", "석사", "박사", "전문박사"]

        base = self.df[self.df['학력'] == "Bachelor's degree"]['salary_mid'].median()
        result = []
        for edu, label in zip(edu_order, edu_labels):
            sub = self.df[self.df['학력'] == edu]
            if len(sub) < 30:
                continue
            med = sub['salary_mid'].median()
            premium = (med / base - 1) * 100
            result.append({
                'edu': label,
                'median': med,
                'premium_pct': premium,
                'n': len(sub),
            })
        return result

    def calculate_market_value(self, job, exp, tech_count):
        """
        시장 가치 계산기
        신입(1년미만) + 기술 1-2개 = 100% 기준
        """
        sub = self.df[self.df['tech_count'] >= 1]  # 0개 제외

        # 직무 필터
        if job != '전체':
            sub_job = sub[sub['job_short'] == job]
        else:
            sub_job = sub

        # 기준값: 해당 직무의 신입 + 1-2개
        base_sub = sub_job[
            (sub_job['코딩 경력(년)'] == '< 1 years') &
            (sub_job['tech_group'] == '1-2개')
        ]
        if len(base_sub) < 5:
            # 표본 부족 시 전체 직무의 신입 + 1-2개 사용
            base_sub = sub[
                (sub['코딩 경력(년)'] == '< 1 years') &
                (sub['tech_group'] == '1-2개')
            ]
        base_med = base_sub['salary_mid'].median()

        # 사용자 조건
        tg = tech_group(tech_count)
        user_sub = sub_job[
            (sub_job['코딩 경력(년)'] == exp) &
            (sub_job['tech_group'] == tg)
        ]

        if len(user_sub) < 5:
            # 기술 구간 완화
            user_sub = sub_job[sub_job['코딩 경력(년)'] == exp]

        if len(user_sub) < 5:
            # 경력만으로
            user_sub = sub_job[sub_job['코딩 경력(년)'] == exp]

        user_med = user_sub['salary_mid'].median()

        if pd.isna(base_med) or pd.isna(user_med) or base_med == 0:
            return {'ratio': 100, 'n': 0}

        ratio = user_med / base_med * 100
        return {
            'ratio': ratio,
            'n': len(user_sub),
            'base_n': len(base_sub),
        }
