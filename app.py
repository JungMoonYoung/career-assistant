import streamlit as st
import os
import pandas as pd
import numpy as np
import json
import random
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from src.rag.rag_pipeline import JobRAGPipeline
from src.indexing.vector_store import JobVectorStore
from src.analysis.kaggle_analyzer import KaggleAnalyzer

# 페이지 설정
st.set_page_config(page_title="나만의 커리어 비서", page_icon=None, layout="wide")

# 세션 상태 초기화 (탭 상태 관리)
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "시장 현황"

# Kaggle 설문 데이터 기반 분석기 로드 (탭2용)
@st.cache_resource
def get_kaggle_analyzer():
    return KaggleAnalyzer()

# 고급 CSS 스타일
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;600;700;900&family=Outfit:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');
    :root {
        --bg-main: #0a0e1a;
        --accent-blue: #638cff;
        --accent-green: #34d399;
        --accent-amber: #fbbf24;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --border: rgba(255, 255, 255, 0.08);
    }
    .stApp {
        background-color: var(--bg-main);
        background-image: 
            radial-gradient(circle at 20% 30%, rgba(99, 140, 255, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 80% 70%, rgba(52, 211, 153, 0.05) 0%, transparent 40%);
        background-attachment: fixed;
    }
    * { font-family: 'Noto Sans KR', sans-serif; }
    h1, h2, h3, .outfit { font-family: 'Outfit', sans-serif; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .header-container { text-align: center; padding: 60px 0 40px 0; animation: fadeDown 0.8s ease-out; }
    .ai-badge {
        display: inline-flex; align-items: center; background: rgba(52, 211, 153, 0.1); 
        color: var(--accent-green); padding: 6px 16px; border-radius: 100px; 
        font-size: 0.85rem; font-weight: 600; margin-bottom: 20px; border: 1px solid rgba(52, 211, 153, 0.2);
    }
    .pulse-dot {
        width: 8px; height: 8px; background: var(--accent-green); border-radius: 50%; 
        margin-right: 8px; position: relative;
    }
    .pulse-dot::after {
        content: ''; position: absolute; width: 100%; height: 100%; background: var(--accent-green); 
        border-radius: 50%; animation: pulse 1.5s infinite;
    }
    .main-title {
        font-size: 4.5rem; font-weight: 900; letter-spacing: -3px; margin: 0;
        background: linear-gradient(135deg, #fff 30%, var(--accent-blue) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sub-title { color: var(--text-muted); font-size: 1.4rem; font-weight: 300; margin-top: 10px; }
    .section-label { display: flex; align-items: center; gap: 12px; margin-bottom: 24px; }
    .section-title { font-size: 1.8rem; font-weight: 700; color: #fff; margin: 0; }
    .tag-badge { background: rgba(251, 191, 36, 0.1); color: var(--accent-amber); padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .source-banner {
        background: rgba(52, 211, 153, 0.05); border: 1px solid rgba(52, 211, 153, 0.1);
        padding: 12px 18px; border-radius: 10px; display: flex; align-items: center; gap: 10px; margin-bottom: 30px;
    }
    .report-item { display: flex; gap: 18px; padding: 16px; border-radius: 12px; transition: 0.25s; margin-bottom: 12px; }
    .report-item:hover { background: rgba(255, 255, 255, 0.03); }
    .badge-num { 
        min-width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; 
        justify-content: center; font-weight: 800; font-size: 0.9rem;
    }
    .prediction-value { 
        font-size: 56px; font-weight: 900; margin: 10px 0;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-green));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .split-bottom { display: grid; grid-template-columns: 1fr 1fr 1fr; border-top: 1px solid var(--border); margin-top: 30px; padding-top: 20px; }
    [data-testid="stHorizontalBlock"] > div:has(div[data-testid="stRadio"]) {
        display: flex !important; justify-content: center !important; width: 100% !important;
    }
    div[data-testid="stRadio"] { display: flex !important; justify-content: center !important; width: 100% !important; }
    div[data-testid="stRadio"] > div {
        background: rgba(17, 24, 39, 0.5) !important; padding: 8px !important; border-radius: 100px !important;
        border: 1px solid var(--border) !important; width: fit-content !important; margin: 0 auto 50px auto !important;
        display: flex !important; gap: 10px !important; justify-content: center !important;
    }
    div[data-testid="stRadio"] label {
        background: transparent !important; padding: 12px 45px !important; border-radius: 100px !important;
        color: var(--text-muted) !important; font-weight: 600 !important; cursor: pointer !important;
        border: none !important; margin: 0 !important; transition: 0.25s !important; font-size: 1.15rem !important;
    }
    div[data-testid="stRadio"] label[aria-checked="true"] {
        background: var(--accent-blue) !important; color: white !important;
        box-shadow: 0 4px 20px rgba(99, 140, 255, 0.5) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child { display: none !important; }
    @keyframes fadeDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
    .stSelectbox div[data-baseweb="select"] { background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 10px; }
    .stMultiSelect div[data-baseweb="select"] { background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 10px; }
    .stButton button { 
        background: linear-gradient(135deg, var(--accent-blue), #4f46e5); color: white; border: none; 
        padding: 12px 24px; border-radius: 12px; font-weight: 700; width: 100%; height: 50px;
        transition: 0.25s; box-shadow: 0 4px 15px rgba(99, 140, 255, 0.3);
    }
    .stButton button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(99, 140, 255, 0.5); color: white; }
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# Kaggle 분석기 로드 (캐싱 사용)
kaggle_analyzer = get_kaggle_analyzer()

# 데이터 로딩 함수 (시장 현황용)
@st.cache_data(ttl=1)
def get_market_data():
    file_path = "data/raw/saramin_10_jobs.json"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f: data = json.load(f)
        rows = []
        for item in data:
            try:
                # 1. 직무명
                job_name = item["position"]["job-mid-code"]["name"]

                # 2. 지역 - location_main(정규화 완료) 사용
                region_name = item.get("location_main", "기타")
                if not region_name:
                    region_name = "기타"

                # 3. 연봉 - salary_number 우선, 없으면 salary.name에서 추출
                salary_str = item.get("salary", {}).get("name", "")
                # 합격축하금은 연봉 데이터에서 제외
                if "축하금" in salary_str:
                    salary_val = 0
                else:
                    salary_val = item.get("salary_number", 0)
                    if not salary_val:
                        import re
                        nums = re.findall(r'[\d,]+', salary_str)
                        if nums:
                            salary_val = int(nums[0].replace(',', ''))
                    # 월급이면 ×12로 연봉 환산
                    if salary_val and "월급" in salary_str:
                        salary_val = salary_val * 12

                # 4. 경력 - career_normalized 우선, 없으면 experience-level에서 추출
                exp_cat = item.get("career_normalized", "")
                if not exp_cat:
                    exp_level = item["position"].get("experience-level", {}).get("name", "")
                    exp_cat = "신입" if "신입" in exp_level else "경력"

                # 5. 출처
                source = item.get("source", "unknown")

                rows.append({
                    "직무": job_name,
                    "지역": region_name,
                    "연봉": salary_val,
                    "경력": exp_cat,
                    "출처": source,
                })
            except Exception:
                continue
        return pd.DataFrame(rows)
    return None

def wrap_region(text):
    if not text: return ""
    text = text.strip()
    return text.replace(" ", "<br>") if " " in text else text

@st.cache_resource(show_spinner=False)
def get_rag_pipeline(): return JobRAGPipeline()

if "messages" not in st.session_state: st.session_state.messages = []

# 헤더 렌더링
st.markdown("<div class='header-container'><div class='ai-badge'><div class='pulse-dot'></div>AI 기반 실시간 분석</div><h1 class='main-title'>나만의 커리어 비서</h1><p class='sub-title'>데이터 기반의 스마트 채용 분석 플랫폼</p></div>", unsafe_allow_html=True)

# 탭 내비게이션
tabs = ["시장 현황", "데이터 분석가 리포트", "커리어 검색"]
st.radio("", tabs, horizontal=True, label_visibility="collapsed", key="active_tab", index=tabs.index(st.session_state.active_tab))

if st.session_state.active_tab == "시장 현황":
    df_market = get_market_data()
    if df_market is not None:
        st.markdown("<div class='section-label'><h2 class='section-title'>전체 채용 시장 현황</h2></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.subheader("직무별 수요 분포")
            # 데이터 요약: '데이터 분석가'는 무조건 포함 + 나머지 중 상위 N개 추출
            raw_counts = df_market["직무"].value_counts().reset_index()
            raw_counts.columns = ["직무", "건수"]
            
            # 1. 데이터 분석가 행 분리
            target_job = "데이터 분석가"
            target_row = raw_counts[raw_counts["직무"] == target_job]
            other_rows = raw_counts[raw_counts["직무"] != target_job]
            
            # 2. 나머지 중 상위 7개만 유지 (데이터 분석가 포함 총 9개 조각 목표)
            top_n = 7 
            top_others = other_rows.head(top_n).copy()
            merged_others_count = other_rows.iloc[top_n:]["건수"].sum()
            
            # 3. 최종 데이터 프레임 합치기
            job_counts = pd.concat([target_row, top_others], ignore_index=True)
            if merged_others_count > 0:
                others_row = pd.DataFrame([{"직무": "기타", "건수": merged_others_count}])
                job_counts = pd.concat([job_counts, others_row], ignore_index=True)

            # 세련된 컬러 팔레트 정의
            distinct_colors = [
                "#1e3a8a", # 데이터 분석가 고정 (Deep Blue)
                "#3b82f6", "#10b981", "#f59e0b", "#ef4444", 
                "#8b5cf6", "#ec4899", "#06b6d4", "#94a3b8" 
            ]

            colors = []
            pulls = []
            for i, job in enumerate(job_counts["직무"]):
                if "데이터 분석가" in job:
                    colors.append("#1e3a8a") # 강조색
                    pulls.append(0.15)      # 밖으로 튀어나오게 (Explode)
                elif job == "기타":
                    colors.append("#cbd5e1") # 연한 회색 (Tone down)
                    pulls.append(0)
                else:
                    # 인덱스에 맞는 뚜렷한 색상 배정
                    colors.append(distinct_colors[i % len(distinct_colors)])
                    pulls.append(0)

            fig_job = go.Figure(data=[go.Pie(
                labels=job_counts["직무"], 
                values=job_counts["건수"], 
                pull=pulls, # 특정 조각 강조 효과
                hole=.6, # 중앙 구멍을 살짝 키움
                marker=dict(colors=colors, line=dict(color='rgba(255,255,255,0.2)', width=1.5)), 
                texttemplate='<b>%{label}</b><br>%{percent}', 
                textposition='inside', 
                insidetextorientation='horizontal', 
                textfont=dict(size=13, color='black'), # 검은색 텍스트
                rotation=90, # 12시 방향부터 시작
                hoverinfo='none'
            )])
            fig_job.update_layout(
                annotations=[],
                paper_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='#f1f5f9'), 
                height=500, # 차트 크기 약간 키움
                margin=dict(t=50, b=50, l=30, r=30), 
                showlegend=False, 
                hovermode=False
            )
            st.plotly_chart(fig_job, use_container_width=True)
        with c2:
            st.subheader("지역별 평균 연봉")
            # 연봉 0 제외, 상위 8개 지역
            df_with_salary = df_market[df_market["연봉"] > 0]
            region_avg = df_with_salary.groupby("지역")["연봉"].agg(["mean", "count"]).reset_index()
            region_avg.columns = ["지역", "연봉", "건수"]
            region_avg = region_avg[(region_avg["건수"] >= 5) & (region_avg["지역"] != "기타")].sort_values("연봉", ascending=False)
            avg_sal = region_avg.head(8).copy()

            # 요약 카드 3개
            top_region = avg_sal.iloc[0]
            total_sal_count = int(avg_sal["건수"].sum())
            overall_avg = (avg_sal["연봉"] * avg_sal["건수"]).sum() / avg_sal["건수"].sum()
            gap = int(avg_sal["연봉"].iloc[0] - avg_sal["연봉"].iloc[-1])

            st.markdown(f"""
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:18px;">
                <div style="background:rgba(83,74,183,0.08); border:1px solid rgba(83,74,183,0.18); border-radius:12px; padding:14px 10px; text-align:center;">
                    <div style="font-size:0.72rem; color:#94a3b8; margin-bottom:6px;">최고 연봉 지역</div>
                    <div style="font-size:1.3rem; font-weight:800; background:linear-gradient(135deg,#AFA9EC,#534AB7); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">{top_region['지역']}</div>
                    <div style="font-size:0.68rem; color:#64748b; margin-top:4px;">평균 {int(top_region['연봉']):,}만원</div>
                </div>
                <div style="background:rgba(83,74,183,0.08); border:1px solid rgba(83,74,183,0.18); border-radius:12px; padding:14px 10px; text-align:center;">
                    <div style="font-size:0.72rem; color:#94a3b8; margin-bottom:6px;">전체 평균 연봉</div>
                    <div style="font-size:1.3rem; font-weight:800; background:linear-gradient(135deg,#AFA9EC,#534AB7); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">{int(overall_avg):,}만원</div>
                    <div style="font-size:0.68rem; color:#64748b; margin-top:4px;">연봉 공개 {total_sal_count:,}건 기준</div>
                </div>
                <div style="background:rgba(83,74,183,0.08); border:1px solid rgba(83,74,183,0.18); border-radius:12px; padding:14px 10px; text-align:center;">
                    <div style="font-size:0.72rem; color:#94a3b8; margin-bottom:6px;">지역 간 격차</div>
                    <div style="font-size:1.3rem; font-weight:800; background:linear-gradient(135deg,#AFA9EC,#534AB7); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">{gap:,}만원</div>
                    <div style="font-size:0.68rem; color:#64748b; margin-top:4px;">1위 vs 최하위</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 색상: 1위 진한 보라, 2위 중간 보라, 나머지 연한 보라
            bar_colors = [('#534AB7' if i == 0 else '#7F77DD' if i == 1 else '#AFA9EC') for i in range(len(avg_sal))]

            fig_sal = go.Figure(go.Bar(
                x=avg_sal["지역"],
                y=avg_sal["연봉"],
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=[f'{int(v):,}' for v in avg_sal["연봉"]],
                textposition='outside',
                textfont=dict(size=13, color='#cbd5e1'),
                cliponaxis=False,
                hoverinfo='none',
            ))
            fig_sal.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8', size=12),
                height=340,
                xaxis=dict(showgrid=False, tickangle=0, showline=False),
                yaxis=dict(title="", showgrid=True, gridcolor='rgba(255,255,255,0.05)', gridwidth=0.5, zeroline=False, showticklabels=False, range=[0, max(avg_sal["연봉"])*1.25], showline=False),
                bargap=0.25, hovermode=False,
                margin=dict(t=35, b=30, l=5, r=5),
            )
            st.plotly_chart(fig_sal, use_container_width=True)
        
        st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2, gap="large")
        with c3:
            st.subheader("경력별 채용 분포")
            exp_counts = df_market["경력"].value_counts().reset_index()
            exp_counts.columns = ["경력", "건수"]
            fig_exp = go.Figure(data=[go.Pie(
                labels=exp_counts["경력"],
                values=exp_counts["건수"],
                hole=.6,
                marker=dict(colors=['#638cff', '#34d399'], line=dict(color='rgba(255,255,255,0.2)', width=1.5)),
                texttemplate='<b>%{label}</b><br>%{percent}',
                textposition='inside',
                insidetextorientation='horizontal',
                textfont=dict(size=14, color='white'),
                hoverinfo='none'
            )])
            fig_exp.update_layout(
                annotations=[],
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f1f5f9'), height=400, margin=dict(t=20, b=20), showlegend=False, hovermode=False
            )
            st.plotly_chart(fig_exp, use_container_width=True)
        with c4:
            st.subheader("연봉 구간별 분포")
            df_sal_only = df_market[df_market["연봉"] > 0].copy()
            bins = [0, 2000, 3000, 4000, 5000, 6000, 7000, 8000, float('inf')]
            bin_labels = ['2천 미만', '2~3천', '3~4천', '4~5천', '5~6천', '6~7천', '7~8천', '8천 이상']
            df_sal_only['연봉구간'] = pd.cut(df_sal_only['연봉'], bins=bins, labels=bin_labels, right=False)
            sal_dist = df_sal_only['연봉구간'].value_counts().sort_index().reset_index()
            sal_dist.columns = ['구간', '건수']
            sal_dist['비율'] = (sal_dist['건수'] / sal_dist['건수'].sum() * 100).round(1)

            # 요약 카드 3개
            top_group = sal_dist.loc[sal_dist['비율'].idxmax(), '구간']
            top_pct = sal_dist['비율'].max()
            mid_mask = sal_dist['구간'].isin(['2~3천', '3~4천', '4~5천'])
            mid_pct = sal_dist.loc[mid_mask, '비율'].sum()
            high_mask = sal_dist['구간'].isin(['6~7천', '7~8천', '8천 이상'])
            high_pct = sal_dist.loc[high_mask, '비율'].sum()

            st.markdown(f"""
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:18px;">
                <div style="background:rgba(83,74,183,0.08); border:1px solid rgba(83,74,183,0.18); border-radius:12px; padding:14px 10px; text-align:center;">
                    <div style="font-size:0.72rem; color:#94a3b8; margin-bottom:6px;">최다 구간</div>
                    <div style="font-size:1.3rem; font-weight:800; background:linear-gradient(135deg,#AFA9EC,#534AB7); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">{top_group}</div>
                    <div style="font-size:0.68rem; color:#64748b; margin-top:4px;">전체의 {top_pct}%</div>
                </div>
                <div style="background:rgba(83,74,183,0.08); border:1px solid rgba(83,74,183,0.18); border-radius:12px; padding:14px 10px; text-align:center;">
                    <div style="font-size:0.72rem; color:#94a3b8; margin-bottom:6px;">중간 구간 집중도</div>
                    <div style="font-size:1.3rem; font-weight:800; background:linear-gradient(135deg,#AFA9EC,#534AB7); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">{mid_pct:.1f}%</div>
                    <div style="font-size:0.68rem; color:#64748b; margin-top:4px;">2천~5천만원 비중</div>
                </div>
                <div style="background:rgba(83,74,183,0.08); border:1px solid rgba(83,74,183,0.18); border-radius:12px; padding:14px 10px; text-align:center;">
                    <div style="font-size:0.72rem; color:#94a3b8; margin-bottom:6px;">6천 이상 비율</div>
                    <div style="font-size:1.3rem; font-weight:800; background:linear-gradient(135deg,#AFA9EC,#534AB7); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">{high_pct:.1f}%</div>
                    <div style="font-size:0.68rem; color:#64748b; margin-top:4px;">고연봉 상위 구간</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 색상: 1위 진한 보라, 2위 중간 보라, 나머지 연한 보라
            DARK_V = '#534AB7'
            MID_V = '#7F77DD'
            LIGHT_V = '#AFA9EC'
            ranked = sal_dist['비율'].sort_values(ascending=False).values
            bar_colors = [DARK_V if v == ranked[0] else MID_V if v == ranked[1] else LIGHT_V for v in sal_dist['비율']]
            # 1% 미만은 라벨 생략
            bar_texts = [f'{v:.1f}%' if v >= 1 else '' for v in sal_dist['비율']]

            fig_hist = go.Figure(go.Bar(
                x=sal_dist['구간'].astype(str),
                y=sal_dist['비율'],
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=bar_texts,
                textposition='outside',
                textfont=dict(size=13, color='#cbd5e1'),
                cliponaxis=False,
                constraintext='none',
                hoverinfo='none',
            ))
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8', size=12),
                height=340,
                xaxis=dict(type='category', showgrid=False, tickangle=0, showline=False),
                yaxis=dict(title="", showgrid=True, gridcolor='rgba(255,255,255,0.05)', gridwidth=0.5, zeroline=False, showticklabels=False, range=[0, max(sal_dist['비율'])*1.3], showline=False),
                bargap=0.25, hovermode=False,
                uniformtext=dict(minsize=11, mode='show'),
                margin=dict(t=35, b=45, l=5, r=5),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

elif st.session_state.active_tab == "데이터 분석가 리포트":
    st.markdown("<div class='section-label'><h2 class='section-title'>데이터 직군 시장 가치 분석</h2></div><div class='source-banner'>Kaggle 글로벌 데이터 직군 설문 데이터 (4,390명) 기반 분석 리포트입니다.</div>", unsafe_allow_html=True)
    col_v1, col_v2 = st.columns([1.4, 1], gap="large")
    with col_v1:
        st.subheader("경력별 연봉 상승 곡선")

        # 직무별 경력 상승 곡선 (Kaggle 실데이터)
        fig_curve = go.Figure()
        job_colors = {
            'Data Analyst': ('#638cff', 'rgba(99, 140, 255, 0.1)'),
            'Data Scientist': ('#34d399', 'rgba(52, 211, 153, 0.1)'),
            'ML Engineer': ('#fbbf24', 'rgba(251, 191, 36, 0.1)'),
            'Data Engineer': ('#f87171', 'rgba(248, 113, 113, 0.1)'),
        }
        job_labels_kr = {
            'Data Analyst': '데이터 분석가',
            'Data Scientist': '데이터 사이언티스트',
            'ML Engineer': 'ML 엔지니어',
            'Data Engineer': '데이터 엔지니어',
        }
        for job_key, (line_color, fill_color) in job_colors.items():
            curve = kaggle_analyzer.get_exp_growth_curve(job_filter=job_key)
            if curve:
                fig_curve.add_trace(go.Scatter(
                    x=[r['exp'] for r in curve],
                    y=[r['step_pct'] for r in curve],
                    name=job_labels_kr.get(job_key, job_key),
                    line=dict(color=line_color, width=4, shape='spline'),
                    fill='tozeroy',
                    fillcolor=fill_color,
                    hoverinfo='none'
                ))
        fig_curve.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9'),
            height=450,
            xaxis=dict(title="경력 단계", showgrid=False),
            yaxis=dict(title="이전 단계 대비 상승률 (%)", gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50),
            hovermode=False
        )
        st.plotly_chart(fig_curve, use_container_width=True)
    with col_v2:
        # 핵심 분석 리포트 (Kaggle 데이터 기반 실제 수치)
        exp_data = kaggle_analyzer.get_exp_growth_curve()
        tech_data = kaggle_analyzer.get_tech_count_premium()
        job_data = kaggle_analyzer.get_job_comparison()

        # 경력 점프가 가장 큰 구간
        if exp_data:
            max_step = max(exp_data[1:], key=lambda x: x['step_pct'])
            step_text = f"{max_step['exp']} 진입 시 <span style='color:#638cff; font-weight:700;'>+{max_step['step_pct']:.0f}%</span> 상승 (가장 큰 점프 구간)"
        else:
            step_text = "데이터 부족"

        # 기술 개수 효과
        if len(tech_data) >= 3:
            tech_text = f"기술 1-2개 대비 6-10개 보유 시 <span style='color:#34d399; font-weight:700;'>+{tech_data[2]['premium_pct']:.0f}%</span> 프리미엄"
        else:
            tech_text = "데이터 부족"

        # 직무별 차이
        if len(job_data) >= 2:
            sorted_jobs = sorted(job_data, key=lambda x: x['median'], reverse=True)
            top_job = sorted_jobs[0]
            bot_job = sorted_jobs[-1]
            gap = (top_job['median'] / bot_job['median'] - 1) * 100
            job_text = f"{top_job['job']}이 {bot_job['job']} 대비 <span style='color:#fbbf24; font-weight:700;'>+{gap:.0f}%</span> 높은 중앙값"
        else:
            job_text = "데이터 부족"

        st.markdown(f"""<div style='min-height: 524px;'>
        <h3 style='margin-bottom:30px;'>핵심 분석 리포트</h3>
        <div class='report-item'><div class='badge-num' style='background:rgba(99,140,255,0.2); color:#638cff;'>1</div><div><div style='font-weight:700; font-size:1.1rem;'>최대 성장 구간</div><div style='color:var(--text-muted); font-size:0.9rem;'>{step_text}</div></div></div>
        <div class='report-item'><div class='badge-num' style='background:rgba(52,211,153,0.2); color:#34d399;'>2</div><div><div style='font-weight:700; font-size:1.1rem;'>기술 스택 효과</div><div style='color:var(--text-muted); font-size:0.9rem;'>{tech_text}</div></div></div>
        <div class='report-item'><div class='badge-num' style='background:rgba(251,191,36,0.2); color:#fbbf24;'>3</div><div><div style='font-weight:700; font-size:1.1rem;'>직무별 격차</div><div style='color:var(--text-muted); font-size:0.9rem;'>{job_text}</div></div></div>
        <div class='report-item'><div class='badge-num' style='background:rgba(255,255,255,0.1); color:#fff;'>4</div><div><div style='font-weight:700; font-size:1.1rem;'>데이터 기반</div><div style='color:var(--text-muted); font-size:0.9rem;'>Kaggle 글로벌 설문 <span style='font-weight:700; color:#fff;'>{kaggle_analyzer.n_total:,}명</span> 실응답 데이터 기반 분석</div></div></div>
        </div>""", unsafe_allow_html=True)

    # === 시장 가치 계산기 ===
    st.markdown("<div class='section-label' style='margin-top:60px;'><h2 class='section-title'>나의 시장 가치 계산기</h2></div>", unsafe_allow_html=True)
    col_p1, col_p2 = st.columns([1, 1.3], gap="large")
    with col_p1:
        st.subheader("나의 조건 입력")
        with st.form("prediction_form", clear_on_submit=False, border=False):
            st.markdown("<p style='font-size:0.9rem; color:var(--text-muted); margin:0 0 5px 0;'>직무</p>", unsafe_allow_html=True)
            p_job = st.selectbox("job", ["Data Analyst", "Data Scientist", "Data Engineer", "ML Engineer"], label_visibility="collapsed")
            st.markdown("<p style='font-size:0.9rem; color:var(--text-muted); margin:15px 0 5px 0;'>코딩 경력</p>", unsafe_allow_html=True)
            exp_options = ["1년 미만", "1-3년", "3-5년", "5-10년", "10-20년", "20년+"]
            exp_keys = ['< 1 years', '1-3 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']
            p_exp = st.selectbox("exp", exp_options, label_visibility="collapsed")
            st.markdown("<p style='font-size:0.9rem; color:var(--text-muted); margin:15px 0 5px 0;'>보유 기술 개수 (언어 + 프레임워크 + 알고리즘)</p>", unsafe_allow_html=True)
            p_tech_count = st.slider("tech_count", min_value=1, max_value=20, value=4, label_visibility="collapsed")
            st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
            if st.form_submit_button("시장 가치 분석하기"): pass

        # 경력 매핑
        exp_key = exp_keys[exp_options.index(p_exp)]

        # Kaggle 데이터 기반 시장 가치 계산
        mv = kaggle_analyzer.calculate_market_value(p_job, exp_key, p_tech_count)
        ratio = mv['ratio']
        change = ratio - 100

    with col_p2:
        # 증감 색상
        if change > 0:
            change_color = "var(--accent-green)"
            change_text = f"+{change:.0f}%"
            change_bg = "rgba(52,211,153,0.1)"
        elif change < 0:
            change_color = "#f87171"
            change_text = f"{change:.0f}%"
            change_bg = "rgba(248,113,113,0.1)"
        else:
            change_color = "var(--text-muted)"
            change_text = "±0%"
            change_bg = "rgba(255,255,255,0.05)"

        from src.analysis.kaggle_analyzer import tech_group as tg_func
        tg_label = tg_func(p_tech_count)

        result_html = f"""<div style='height:100%;'>
        <p style='color: var(--text-muted); font-size: 1.1rem; margin-bottom: 0;'>신입(1년미만) + 기술 1-2개 대비</p>
        <div class='prediction-value outfit'>{change_text}</div>
        <div style='display:inline-block; background:{change_bg}; color:{change_color}; padding:4px 12px; border-radius:100px; font-size:0.9rem; font-weight:700;'>시장 가치 비율 {ratio:.0f}%</div>
        <div class='split-bottom'>
            <div style='text-align:center;'>
                <div style='color:var(--text-muted); font-size:0.8rem;'>직무</div>
                <div class='mono' style='font-size:1.0rem; font-weight:600;'>{p_job}</div>
            </div>
            <div style='text-align:center; border-left:1px solid var(--border); border-right:1px solid var(--border);'>
                <div style='color:var(--accent-blue); font-size:0.8rem;'>경력</div>
                <div class='mono' style='font-size:1.0rem; font-weight:600; color:var(--accent-blue);'>{p_exp}</div>
            </div>
            <div style='text-align:center;'>
                <div style='color:var(--text-muted); font-size:0.8rem;'>기술 스택</div>
                <div class='mono' style='font-size:1.0rem; font-weight:600;'>{p_tech_count}개 ({tg_label})</div>
            </div>
        </div>
        <div style='margin-top:20px; display: flex; gap: 30px; color: var(--text-muted); font-size: 0.8rem;'>
            <span>분석 표본: <b>{mv['n']}명</b></span>
            <span>데이터: <b>Kaggle 설문 {kaggle_analyzer.n_total:,}명</b></span>
        </div>
        </div>"""
        st.markdown(result_html, unsafe_allow_html=True)

elif st.session_state.active_tab == "커리어 검색":
    st.markdown("<div class='section-label'><h2 class='section-title'>AI 커리어 검색</h2></div>", unsafe_allow_html=True)
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
    st.markdown("<p style='color: var(--text-muted); margin-bottom: 5px; font-size: 0.9rem;'>직무와 조건을 검색해 주세요</p>", unsafe_allow_html=True)
    if prompt := st.chat_input("예: 강남 데이터 분석가 신입"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("AI 컨설턴트가 분석 중입니다..."):
                    answer = get_rag_pipeline().ask(prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

st.empty()
