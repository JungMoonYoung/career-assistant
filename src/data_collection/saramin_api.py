import requests
import os
import json
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv(override=True)

class SaraminAPI:
    """사람인 API를 통한 실제 공고 데이터 수집 클래스"""
    
    def __init__(self):
        self.access_key = os.getenv("SARAMIN_ACCESS_KEY")
        self.base_url = "https://oapi.saramin.co.kr/job-search"

    def fetch_and_save(self, keywords: str = "파이썬", count: int = 50):
        """실제 데이터를 가져와서 json 파일로 저장합니다."""
        if not self.access_key or "your_saramin" in self.access_key:
            print("❌ 오류: SARAMIN_ACCESS_KEY가 설정되지 않았습니다.")
            return

        params = {
            "access-key": self.access_key,
            "keywords": keywords,
            "count": count,
            "fields": "expiration-date,count"
        }
        
        try:
            print(f"🔍 사람인에서 '{keywords}' 관련 공고 {count}개를 수집 중...")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            jobs = data.get("jobs", {}).get("job", [])
            
            # 수집된 데이터를 파일로 저장
            save_path = "data/raw/saramin_jobs.json"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(jobs, f, ensure_ascii=False, indent=4)
                
            print(f"✅ 수집 완료: {len(jobs)}개의 공고가 '{save_path}'에 저장되었습니다.")
            return jobs
        except Exception as e:
            print(f"❌ 사람인 데이터 수집 실패: {e}")
            return []

if __name__ == "__main__":
    saramin = SaraminAPI()
    # 파이썬, 자바, 신입 등 원하는 키워드로 수집 가능
    saramin.fetch_and_save(keywords="파이썬 개발자", count=30)
