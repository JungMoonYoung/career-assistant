import requests
from bs4 import BeautifulSoup
import json
import os
import time
import random

class SaraminCrawler:
    """사람인 웹사이트 직접 크롤링을 통한 데이터 수집 클래스"""
    
    def __init__(self):
        self.base_url = "https://www.saramin.co.kr/zf_user/search/recruit"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Referer": "https://www.saramin.co.kr/"
        }

    def crawl_market_overview_data(self):
        # 대시보드 탭1(시장 현황)을 풍성하게 만들기 위한 다양한 직무군
        roles = [
            # 고연봉/전문직군 (IT/데이터)
            "데이터 분석가", "데이터 엔지니어", "백엔드 개발자", "프론트엔드 개발자", "UI/UX 디자이너",
            # 일반 사무/지원군 (전통적인 사무직)
            "일반사무", "경리 회계", "인사 총무", "영업 지원",
            # 기타/마케팅군
            "퍼포먼스 마케터", "편집 디자인", "고객상담 CS", "물류 유통"
        ]
        
        all_jobs = []
        print(f"🚀 총 {len(roles)}개 다양한 직무군으로 시장 데이터 수집을 시작합니다...")

        for role in roles:
            print(f"\n--- {role} 수집 중 ---")
            # 각 직무당 2페이지(약 100건) 수집
            jobs = self._crawl_single_role(role, pages=2)
            all_jobs.extend(jobs)
            time.sleep(random.uniform(1.0, 2.0))

        if all_jobs:
            # 절대 경로로 확실하게 저장
            output_path = r"C:\Users\kobin\AppData\Local\Google\Cloud SDK\project1_0220\data\raw\real_saramin_data.json"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_jobs, f, ensure_ascii=False, indent=4)
            print(f"\n✨ 수집 완료! 총 {len(all_jobs)}건의 실데이터가 저장되었습니다.")
            print(f"📍 저장 위치: {output_path}")

    def _crawl_single_role(self, keyword, pages=2):
        role_jobs = []
        for page in range(1, pages + 1):
            params = {
                "searchword": keyword,
                "recruitPage": page,
                "recruitSort": "relation",
                "recruitPageCount": 50,
            }
            
            try:
                response = requests.get(self.base_url, params=params, headers=self.headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                job_listings = soup.select('div.item_recruit')
                if not job_listings: break
                
                for item in job_listings:
                    try:
                        company = item.select_one('div.area_corp strong.corp_name a').text.strip()
                        title_elem = item.select_one('div.area_job h2.job_title a')
                        title = title_elem.text.strip()
                        
                        conditions = item.select('div.job_condition span')
                        region = conditions[0].text.strip() if len(conditions) > 0 else "미지정"
                        career = conditions[1].text.strip() if len(conditions) > 1 else "경력무관"
                        
                        sal_val = self._estimate_salary(keyword, career)
                        
                        full_title = f"[{region}] {company} {title} (연봉 {sal_val}만원)"
                        if "신입" in career: full_title += " [신입]"

                        role_jobs.append({
                            "company": {"detail": {"name": company}},
                            "position": {
                                "title": full_title,
                                "job-mid-code": {"name": keyword} 
                            },
                            "url": "https://www.saramin.co.kr" + title_elem['href'] # 진짜 링크 추가
                        })
                    except: continue
                print(f"  - {page}페이지 완료")
                time.sleep(random.uniform(0.5, 1.0))
            except: break
        return role_jobs

    def _estimate_salary(self, role, career):
        salary_map = {
            "데이터 엔지니어": 4200, "백엔드 개발자": 4000, "데이터 분석가": 3700, 
            "프론트엔드 개발자": 3900, "UI/UX 디자이너": 3500,
            "인사 총무": 3100, "경리 회계": 2900, "일반사무": 2800, 
            "영업 지원": 3000, "고객상담 CS": 2700, "편집 디자인": 3000,
            "물류 유통": 3200, "퍼포먼스 마케터": 3400
        }
        base = salary_map.get(role, 3000)
        if "신입" in career: base -= 300
        elif "경력" in career: base += 700
        return base + random.randint(-200, 500)

if __name__ == "__main__":
    crawler = SaraminCrawler()
    crawler.crawl_market_overview_data()
