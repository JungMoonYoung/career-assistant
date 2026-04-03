"""
잡코리아 추가 크롤링 - 연봉 표본 200건 목표
기존 데이터에 추가 병합
"""
import sys
import os
import json
import time
import random
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

PROGRESS_FILE = "data/raw/_crawl_progress2.txt"

def log_progress(msg):
    print(msg, flush=True)
    try:
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            f.flush()
    except:
        pass


# 직무별 추가 크롤링 필요량 (연봉 게재율 고려, 여유분 포함)
ADDITIONAL_TARGETS = {
    "영업 지원":         987,
    "데이터 분석가":      812,
    "UI/UX 디자이너":    631,
    "데이터 엔지니어":    626,
    "프론트엔드 개발자":   574,
    "인사 총무":         453,
    "퍼포먼스 마케터":    392,
    "백엔드 개발자":      373,
    "경리 회계":         344,
    "고객상담 CS":       314,
    "일반사무":          166,
}


class JobkoreaCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;"
                      "q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.jobkorea.co.kr/",
        })
        self.request_count = 0
        self.start_time = time.time()

    def safe_get(self, url, params=None):
        delay = random.uniform(3.0, 5.0)
        time.sleep(delay)

        try:
            resp = self.session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            self.request_count += 1

            if self.request_count % 20 == 0:
                rest = random.uniform(10.0, 15.0)
                elapsed = (time.time() - self.start_time) / 60
                log_progress(f"    [안전] {self.request_count}번째 요청, "
                             f"경과: {elapsed:.1f}분, {rest:.0f}초 휴식...")
                time.sleep(rest)

            return resp
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                log_progress(f"    [경고] 429 감지! 60초 대기...")
                time.sleep(60)
                return self.safe_get(url, params)
            elif resp.status_code == 403:
                log_progress(f"    [경고] 403 차단. 중단.")
                return None
            else:
                log_progress(f"    [오류] HTTP {resp.status_code}")
                return None
        except Exception as e:
            log_progress(f"    [오류] {e}")
            return None

    def crawl_role(self, keyword, target_count):
        all_jobs = []
        page = 1
        max_pages = (target_count // 20) + 5
        empty_count = 0

        while len(all_jobs) < target_count and page <= max_pages:
            params = {
                "stext": keyword,
                "tabType": "recruit",
                "Page_No": page,
            }

            resp = self.safe_get("https://www.jobkorea.co.kr/Search/", params)
            if resp is None:
                break

            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, 'html.parser')
            cards = soup.select('.dlua7o0')

            if not cards:
                empty_count += 1
                if empty_count >= 2:
                    break
                page += 1
                continue

            for card in cards:
                if len(all_jobs) >= target_count:
                    break
                job = self._parse_card(card, keyword)
                if job:
                    all_jobs.append(job)

            page += 1
            time.sleep(random.uniform(2.0, 4.0))

        return all_jobs

    def _parse_card(self, card, search_keyword):
        try:
            spans = card.select('span')
            texts = [s.text.strip() for s in spans if s.text.strip()]

            if len(texts) < 4:
                return None

            if texts[0] == '스크랩':
                texts = texts[1:]

            title = texts[0] if len(texts) > 0 else ""
            company = texts[1] if len(texts) > 1 else ""
            region = texts[2] if len(texts) > 2 else ""

            salary_str = ""
            career_str = ""
            for t in texts:
                if '만원' in t or '억원' in t:
                    salary_str = t
                if '경력' in t or '신입' in t:
                    career_str = t

            link = card.select_one('a[href*="Recruit"]')
            url = ""
            if link:
                href = link.get('href', '')
                url = href if href.startswith('http') else "https://www.jobkorea.co.kr" + href

            region_main = "기타"
            for r in ['서울', '경기', '인천', '부산', '대구', '대전', '광주',
                       '울산', '세종', '강원', '충북', '충남', '전북', '전남',
                       '경북', '경남', '제주']:
                if r in region:
                    region_main = r
                    break

            if "신입" in career_str and "경력" in career_str:
                career_normalized = "신입/경력"
            elif "신입" in career_str:
                career_normalized = "신입"
            elif "경력" in career_str:
                career_normalized = "경력"
            else:
                career_normalized = "경력무관"

            salary_number = 0
            if salary_str:
                try:
                    numbers = re.findall(r'[\d,]+', salary_str)
                    if numbers:
                        salary_number = int(numbers[0].replace(',', ''))
                except:
                    pass

            return {
                "source": "jobkorea",
                "company": {"detail": {"name": company}},
                "position": {
                    "title": title,
                    "job-mid-code": {"name": search_keyword},
                    "experience-level": {"name": career_str},
                    "required-education-level": {"name": ""}
                },
                "salary": {"name": salary_str},
                "salary_number": salary_number,
                "location": {"name": region},
                "location_main": region_main,
                "career_normalized": career_normalized,
                "url": url
            }
        except:
            return None


def run():
    log_progress("=" * 70)
    log_progress("  잡코리아 추가 크롤링 (연봉 표본 200건 목표)")
    log_progress(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_progress("=" * 70)

    crawler = JobkoreaCrawler()
    additional_data = []
    role_stats = {}

    roles = list(ADDITIONAL_TARGETS.items())
    random.shuffle(roles)

    for idx, (role, target) in enumerate(roles, 1):
        log_progress(f"\n[{idx}/{len(roles)}] {role} (추가 목표: {target}건)")
        log_progress("-" * 50)

        jobs = crawler.crawl_role(role, target)
        with_salary = sum(1 for j in jobs if j.get('salary_number', 0) > 0)

        role_stats[role] = {
            "crawled": len(jobs),
            "with_salary": with_salary
        }
        additional_data.extend(jobs)

        log_progress(f"  >> {len(jobs)}건 수집 (연봉 보유: {with_salary}건), 누적: {len(additional_data)}건")

        if idx < len(roles):
            between = random.uniform(5.0, 10.0)
            log_progress(f"  [대기] {between:.0f}초...")
            time.sleep(between)

    # 기존 데이터 로드 및 병합
    log_progress(f"\n{'=' * 70}")
    log_progress("  병합 중...")

    existing_path = "data/raw/crawled_jobs_all.json"
    with open(existing_path, "r", encoding="utf-8") as f:
        existing_data = json.load(f)

    merged = existing_data + additional_data
    random.shuffle(merged)

    # 저장
    with open(existing_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    compat_path = "data/raw/saramin_10_jobs.json"
    with open(compat_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # 최종 연봉 표본 집계
    from collections import defaultdict
    final_salary = defaultdict(int)
    for d in merged:
        if d.get('salary_number', 0) > 0 or (d.get('source') == 'jobkorea' and d.get('salary', {}).get('name', '')):
            role = d['position']['job-mid-code']['name']
            if d.get('salary_number', 0) > 0:
                final_salary[role] += 1

    elapsed = (time.time() - crawler.start_time) / 60
    log_progress(f"\n{'=' * 70}")
    log_progress(f"  추가 크롤링 완료!")
    log_progress(f"  소요 시간: {elapsed:.1f}분")
    log_progress(f"  추가 수집: {len(additional_data)}건")
    log_progress(f"  총 데이터: {len(merged)}건")
    log_progress(f"  요청 횟수: {crawler.request_count}회")

    log_progress(f"\n  직무별 추가 수집:")
    for role, stats in sorted(role_stats.items(), key=lambda x: x[1]['with_salary'], reverse=True):
        log_progress(f"    {role:15s}: {stats['crawled']:>5}건 크롤링, 연봉 {stats['with_salary']:>4}건")

    log_progress(f"\n  최종 연봉 표본 (전체 데이터 기준):")
    for role in sorted(final_salary.keys(), key=lambda x: final_salary[x], reverse=True):
        status = "OK" if final_salary[role] >= 200 else "부족"
        log_progress(f"    {role:15s}: {final_salary[role]:>4}건  [{status}]")

    log_progress(f"\n  완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_progress("=" * 70)


if __name__ == "__main__":
    run()
