import re
import pandas as pd
from typing import List, Dict
from langchain_core.documents import Document

class JobPreprocessor:
    """분석가 관점의 고도화된 데이터 전처리 클래스"""
    
    def __init__(self):
        # 분석에 필요한 핵심 기술 스택 리스트 (확장 가능)
        self.tech_keywords = ['Python', 'SQL', 'R', 'Tableau', 'PowerBI', 'Spark', 'Hadoop', 
                              'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Java', 'React', 'Next.js']

    def clean_text(self, text: str) -> str:
        """HTML 태그 제거 및 텍스트 정규화"""
        if not text: return ""
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_skills(self, text: str) -> List[str]:
        """텍스트에서 미리 정의된 기술 스택 키워드 추출 (Feature Engineering)"""
        found_skills = []
        for skill in self.tech_keywords:
            if re.search(rf'\b{skill}\b', text, re.IGNORECASE):
                found_skills.append(skill)
        return found_skills

    def convert_to_documents(self, raw_jobs: List[Dict], source_name: str) -> List[Document]:
        """분석 가능한 구조화된 Document 객체로 변환"""
        documents = []
        
        for job in raw_jobs:
            title = job.get("position", {}).get("title", "")
            desc = job.get("position", {}).get("job-mid-code", {}).get("name", "")
            full_text = f"{title} {desc}"
            
            # 1. 기술 스택 추출 (분석 핵심)
            skills = self.extract_skills(full_text)
            
            # 2. 연봉 데이터 추출 및 정규화
            salary_match = re.search(r'연봉 (\d+)', title)
            salary = int(salary_match.group(1)) if salary_match else 0
            
            # 3. 경력 구간화 (Categorization)
            if "신입" in title: exp_cat = "신입"
            elif "3년차" in title: exp_cat = "주니어(3-5년)"
            elif "8년차" in title: exp_cat = "시니어(8-10년)"
            else: exp_cat = "경력무관"

            content = self.clean_text(full_text)
            
            metadata = {
                "company": job.get("company", {}).get("detail", {}).get("name", "알 수 없음"),
                "title": title,
                "salary": salary,
                "skills": ", ".join(skills),
                "exp_category": exp_cat,
                "url": job.get("url", "")
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
            
        return documents
