import requests
import os
import json
import xml.etree.ElementTree as ET
import time
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class WorknetAPI:
    """전체 채용 시장 데이터를 위한 고용24(워크넷) API 수집 클래스"""
    
    def __init__(self):
        self.auth_key = os.getenv("WORKNET_AUTH_KEY")
        # 고용24 통합 API 새 주소
        self.base_url = "https://www.work24.go.kr/cm/openApi/call/wk/callOpenApiSvcInfo210L01.do"

    def fetch_all_jobs(self, total_count: int = 1000):
        if not self.auth_key:
            print("오류: WORKNET_AUTH_KEY가 설정되어 있지 않습니다.")
            return

        all_processed_jobs = []
        pages = (total_count // 100) + (1 if total_count % 100 != 0 else 0)
        
        print(f"고용24(워크넷) 실시간 채용 데이터 수집 시작 (목표: {total_count}건)...")
        
        for page in range(1, pages + 1):
            params = {
                "authKey": self.auth_key,
                "callTp": "L",
                "returnType": "XML",
                "startPage": page,
                "display": 100
            }
            
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                
                # 에러 체크
                msg_cd = root.findtext(".//messageCd")
                if msg_cd and msg_cd != "000":
                    msg = root.findtext(".//message")
                    print(f"API 에러 ({msg_cd}): {msg}")
                    break
                
                raw_jobs = root.findall(".//wanted")
                if not raw_jobs:
                    break
                
                for wanted in raw_jobs:
                    company = wanted.findtext("company")
                    title_text = wanted.findtext("title")
                    region = wanted.findtext("region")
                    sal = wanted.findtext("sal")
                    career = wanted.findtext("career")
                    jobs_nm = wanted.findtext("jobsNm")
                    
                    # 연봉 수치화 (워크넷은 "월급 300만원", "연봉 4000만원" 등 혼재)
                    # 여기서는 간단한 파싱 로직 적용
                    sal_val = 3000 # 기본값
                    if sal and "연봉" in sal:
                        try:
                            sal_val = int(''.join(filter(str.isdigit, sal.split("~")[0])))
                        except: pass
                    elif sal and "월급" in sal:
                        try:
                            sal_val = int(''.join(filter(str.isdigit, sal.split("~")[0]))) * 12 // 10000 # 월급 -> 연봉(만원)
                        except: pass

                    full_title = f"[{region}] {company} {title_text} 채용 (연봉 {sal_val}만원)"
                    if "신입" in career: full_title += " [신입]"
                    
                    all_processed_jobs.append({
                        "position": {
                            "title": full_title,
                            "job-mid-code": {"name": jobs_nm if jobs_nm else "기타"}
                        },
                        "source": "Worknet"
                    })
                
                print(f"현재 {len(all_processed_jobs)}건 수집 완료...")
                time.sleep(0.5) # API 부하 방지

            except Exception as e:
                print(f"페이지 {page} 수집 중 오류: {e}")
                break
        
        if all_processed_jobs:
            output_path = "data/raw/large_sample_jobs.json"
            os.makedirs("data/raw", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_processed_jobs, f, ensure_ascii=False, indent=4)
            print(f"성공: 총 {len(all_processed_jobs)}건의 실시간 공고가 {output_path}에 저장되었습니다.")
        else:
            print("수집된 데이터가 없습니다. 인증키 활성화를 기다려주세요.")

if __name__ == "__main__":
    worknet = WorknetAPI()
    worknet.fetch_all_jobs(total_count=1000)
