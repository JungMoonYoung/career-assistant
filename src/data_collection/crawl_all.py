"""
사람인 + 잡코리아 통합 크롤러
목표: ~5,000건, 직무별 불균등 분포 (실제 시장 반영)
탭1(시장 현황) + 탭3(커리어 검색) 공용 데이터

안전 수칙:
- 요청 간 3~5초 랜덤 딜레이
- 페이지 전환 시 5~8초 딜레이
- 정상 브라우저 User-Agent
- 과도한 동시 요청 없음 (순차 처리)
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

# 진행률 로그 파일 (실시간 확인용)
PROGRESS_FILE = "data/raw/_crawl_progress.txt"

def log_progress(msg):
    """콘솔 + 파일 동시 로그"""
    print(msg, flush=True)
    try:
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            f.flush()
    except:
        pass


# ============================================================
# 직무별 목표 건수 (불균등 분포, 합산 ~5,000건)
# 실제 시장에서 IT/개발 공고가 많고, CS/사무는 적은 패턴 반영
# ============================================================
ROLE_TARGETS = {
    "데이터 분석가":     600,   # IT/데이터 - 핵심 직무
    "데이터 엔지니어":   500,   # IT/데이터
    "백엔드 개발자":     700,   # 개발 - 가장 수요 많음
    "프론트엔드 개발자": 550,   # 개발
    "UI/UX 디자이너":   400,   # 디자인
    "일반사무":          450,   # 사무
    "경리 회계":         350,   # 사무
    "인사 총무":         350,   # 사무
    "퍼포먼스 마케터":   400,   # 마케팅
    "영업 지원":         350,   # 영업
    "고객상담 CS":       350,   # 서비스
}
# 합계: 5,000건

# 사람인 60%, 잡코리아 40% 비율로 분배
SARAMIN_RATIO = 0.6
JOBKOREA_RATIO = 0.4


class SafeCrawler:
    """안전한 크롤링을 위한 기본 클래스"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;"
                      "q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        })
        self.request_count = 0
        self.start_time = None

    def safe_get(self, url, params=None, referer=None):
        """안전한 GET 요청 (딜레이 + 에러 핸들링)"""
        if self.start_time is None:
            self.start_time = time.time()

        # 요청 간 딜레이 (3~5초)
        delay = random.uniform(3.0, 5.0)
        time.sleep(delay)

        if referer:
            self.session.headers["Referer"] = referer

        try:
            resp = self.session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            self.request_count += 1

            # 매 20번 요청마다 긴 휴식 (10~15초)
            if self.request_count % 20 == 0:
                rest = random.uniform(10.0, 15.0)
                elapsed = time.time() - self.start_time
                log_progress(f"    [안전] {self.request_count}번째 요청 완료, "
                      f"경과: {elapsed/60:.1f}분, {rest:.0f}초 휴식...")
                time.sleep(rest)

            return resp
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                log_progress(f"    [경고] 요청 제한(429) 감지! 60초 대기 후 재시도...")
                time.sleep(60)
                return self.safe_get(url, params, referer)
            elif resp.status_code == 403:
                log_progress(f"    [경고] 접근 차단(403). 이 사이트 크롤링 중단.")
                return None
            else:
                log_progress(f"    [오류] HTTP {resp.status_code}: {e}")
                return None
        except Exception as e:
            log_progress(f"    [오류] 요청 실패: {e}")
            return None


class SaraminCrawler(SafeCrawler):
    """사람인 크롤러"""

    def __init__(self):
        super().__init__()
        self.base_url = "https://www.saramin.co.kr/zf_user/search/recruit"
        self.session.headers["Referer"] = "https://www.saramin.co.kr/"

    def crawl_role(self, keyword, target_count):
        """특정 직무를 목표 건수만큼 크롤링"""
        all_jobs = []
        page = 1
        max_pages = (target_count // 30) + 3  # 여유 페이지
        empty_count = 0

        while len(all_jobs) < target_count and page <= max_pages:
            params = {
                "searchword": keyword,
                "recruitPage": page,
                "recruitSort": "relation",
                "recruitPageCount": 50,
            }

            resp = self.safe_get(self.base_url, params=params,
                                 referer="https://www.saramin.co.kr/")
            if resp is None:
                break

            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, 'html.parser')
            listings = soup.select('div.item_recruit')

            if not listings:
                empty_count += 1
                if empty_count >= 2:
                    break
                page += 1
                continue

            for item in listings:
                if len(all_jobs) >= target_count:
                    break
                job = self._parse_saramin_item(item, keyword)
                if job:
                    all_jobs.append(job)

            page += 1

            # 페이지 간 추가 딜레이
            time.sleep(random.uniform(2.0, 4.0))

        return all_jobs

    def _parse_saramin_item(self, item, search_keyword):
        """사람인 개별 공고 파싱"""
        try:
            company_elem = item.select_one('strong.corp_name a')
            title_elem = item.select_one('h2.job_tit a')

            if not company_elem or not title_elem:
                return None

            company = company_elem.text.strip()
            title = title_elem.text.strip()
            href = title_elem.get('href', '')

            conditions = item.select('div.job_condition span')
            region = conditions[0].text.strip() if len(conditions) > 0 else ""
            career = conditions[1].text.strip() if len(conditions) > 1 else ""
            education = conditions[2].text.strip() if len(conditions) > 2 else ""
            employment_type = conditions[3].text.strip() if len(conditions) > 3 else ""

            # 지역 정규화
            region_main = self._normalize_region(region)

            # 경력 정규화
            career_normalized = self._normalize_career(career)

            return {
                "source": "saramin",
                "company": {"detail": {"name": company}},
                "position": {
                    "title": title,
                    "job-mid-code": {"name": search_keyword},
                    "experience-level": {"name": career},
                    "required-education-level": {"name": education}
                },
                "salary": {"name": ""},  # 사람인은 연봉 미표기가 대부분
                "location": {"name": region},
                "location_main": region_main,
                "career_normalized": career_normalized,
                "employment_type": employment_type,
                "url": ("https://www.saramin.co.kr" + href) if href else ""
            }
        except Exception:
            return None

    def _normalize_region(self, region_str):
        """지역 정규화: '서울 강남구' → '서울'"""
        for r in ['서울', '경기', '인천', '부산', '대구', '대전', '광주',
                   '울산', '세종', '강원', '충북', '충남', '전북', '전남',
                   '경북', '경남', '제주']:
            if r in region_str:
                return r
        return "기타"

    def _normalize_career(self, career_str):
        """경력 정규화"""
        if "신입" in career_str and "경력" in career_str:
            return "신입/경력"
        elif "신입" in career_str:
            return "신입"
        elif "경력" in career_str:
            return "경력"
        else:
            return "경력무관"


class JobkoreaCrawler(SafeCrawler):
    """잡코리아 크롤러"""

    def __init__(self):
        super().__init__()
        self.base_url = "https://www.jobkorea.co.kr/Search/"
        self.session.headers["Referer"] = "https://www.jobkorea.co.kr/"

    def crawl_role(self, keyword, target_count):
        """특정 직무를 목표 건수만큼 크롤링"""
        all_jobs = []
        page = 1
        max_pages = (target_count // 20) + 3
        empty_count = 0

        while len(all_jobs) < target_count and page <= max_pages:
            params = {
                "stext": keyword,
                "tabType": "recruit",
                "Page_No": page,
            }

            resp = self.safe_get(self.base_url, params=params,
                                 referer="https://www.jobkorea.co.kr/")
            if resp is None:
                break

            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, 'html.parser')

            # 잡코리아 공고 카드
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
                job = self._parse_jobkorea_card(card, keyword)
                if job:
                    all_jobs.append(job)

            page += 1
            time.sleep(random.uniform(2.0, 4.0))

        return all_jobs

    def _parse_jobkorea_card(self, card, search_keyword):
        """잡코리아 카드 파싱"""
        try:
            spans = card.select('span')
            texts = [s.text.strip() for s in spans if s.text.strip()]

            # texts 구조: [스크랩, 제목, 회사명, 지역, 직무분야, 연봉, ...]
            if len(texts) < 4:
                return None

            # '스크랩' 텍스트 제거
            if texts[0] == '스크랩':
                texts = texts[1:]

            title = texts[0] if len(texts) > 0 else ""
            company = texts[1] if len(texts) > 1 else ""
            region = texts[2] if len(texts) > 2 else ""

            # 연봉 찾기 (연봉/월급 패턴)
            salary_str = ""
            career_str = ""
            for t in texts:
                if '만원' in t or '억원' in t:
                    salary_str = t
                if '경력' in t or '신입' in t:
                    career_str = t

            # URL 찾기
            link = card.select_one('a[href*="Recruit"]')
            url = ""
            if link:
                href = link.get('href', '')
                if href.startswith('http'):
                    url = href
                else:
                    url = "https://www.jobkorea.co.kr" + href

            # 지역 정규화
            region_main = self._normalize_region(region)

            # 경력 정규화
            career_normalized = self._normalize_career(career_str)

            # 연봉 숫자 추출
            salary_normalized = self._normalize_salary(salary_str)

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
                "salary_number": salary_normalized,
                "location": {"name": region},
                "location_main": region_main,
                "career_normalized": career_normalized,
                "url": url
            }
        except Exception:
            return None

    def _normalize_region(self, region_str):
        for r in ['서울', '경기', '인천', '부산', '대구', '대전', '광주',
                   '울산', '세종', '강원', '충북', '충남', '전북', '전남',
                   '경북', '경남', '제주']:
            if r in region_str:
                return r
        return "기타"

    def _normalize_career(self, career_str):
        if "신입" in career_str and "경력" in career_str:
            return "신입/경력"
        elif "신입" in career_str:
            return "신입"
        elif "경력" in career_str:
            return "경력"
        else:
            return "경력무관"

    def _normalize_salary(self, salary_str):
        """연봉 문자열 → 만원 단위 숫자"""
        if not salary_str:
            return 0
        try:
            # "연봉 5,000만원~" → 5000
            numbers = re.findall(r'[\d,]+', salary_str)
            if numbers:
                val = int(numbers[0].replace(',', ''))
                if '억' in salary_str:
                    val = val * 10000
                return val
        except:
            pass
        return 0


def run_crawling():
    """메인 크롤링 실행"""
    log_progress("=" * 70)
    log_progress("  사람인 + 잡코리아 통합 크롤러")
    log_progress(f"  목표: ~{sum(ROLE_TARGETS.values()):,}건 (직무별 불균등 분포)")
    log_progress(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_progress("=" * 70)

    saramin = SaraminCrawler()
    jobkorea = JobkoreaCrawler()

    all_data = []
    role_stats = {}

    roles = list(ROLE_TARGETS.items())
    random.shuffle(roles)  # 직무 순서도 랜덤

    for idx, (role, target) in enumerate(roles, 1):
        saramin_target = int(target * SARAMIN_RATIO)
        jobkorea_target = target - saramin_target

        log_progress(f"\n[{idx}/{len(roles)}] {role} (목표: {target}건 = "
              f"사람인 {saramin_target} + 잡코리아 {jobkorea_target})")
        log_progress("-" * 50)

        # 사람인 크롤링
        log_progress(f"  [사람인] 크롤링 시작...")
        saramin_jobs = saramin.crawl_role(role, saramin_target)
        log_progress(f"  [사람인] {len(saramin_jobs)}건 수집 완료")

        # 사이트 전환 딜레이 (5~8초)
        switch_delay = random.uniform(5.0, 8.0)
        log_progress(f"  [대기] 사이트 전환 {switch_delay:.0f}초...")
        time.sleep(switch_delay)

        # 잡코리아 크롤링
        log_progress(f"  [잡코리아] 크롤링 시작...")
        jobkorea_jobs = jobkorea.crawl_role(role, jobkorea_target)
        log_progress(f"  [잡코리아] {len(jobkorea_jobs)}건 수집 완료")

        role_total = len(saramin_jobs) + len(jobkorea_jobs)
        role_stats[role] = {
            "saramin": len(saramin_jobs),
            "jobkorea": len(jobkorea_jobs),
            "total": role_total
        }

        all_data.extend(saramin_jobs)
        all_data.extend(jobkorea_jobs)

        log_progress(f"  >> {role} 합계: {role_total}건 (누적: {len(all_data)}건)")

        # 직무 간 딜레이 (5~10초)
        if idx < len(roles):
            between_delay = random.uniform(5.0, 10.0)
            log_progress(f"  [대기] 다음 직무까지 {between_delay:.0f}초...")
            time.sleep(between_delay)

    # 결과 저장
    log_progress(f"\n{'=' * 70}")
    log_progress("  크롤링 완료! 저장 중...")
    log_progress(f"{'=' * 70}")

    # 데이터 셔플 (사이트별로 뭉치지 않도록)
    random.shuffle(all_data)

    output_path = "data/raw/crawled_jobs_all.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    # 탭1/탭3 호환용으로도 저장 (기존 파일명)
    compat_path = "data/raw/saramin_10_jobs.json"
    with open(compat_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    # 통계 출력
    elapsed = (time.time() - saramin.start_time) / 60 if saramin.start_time else 0
    log_progress(f"\n  총 소요 시간: {elapsed:.1f}분")
    log_progress(f"  총 수집 건수: {len(all_data)}건")
    log_progress(f"  총 요청 횟수: 사람인 {saramin.request_count}회 + "
          f"잡코리아 {jobkorea.request_count}회")
    log_progress(f"\n  직무별 수집 현황:")

    for role, stats in sorted(role_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        log_progress(f"    {role:15s}: {stats['total']:>4}건 "
              f"(사람인 {stats['saramin']:>3} + 잡코리아 {stats['jobkorea']:>3})")

    # 잡코리아 연봉 데이터 통계
    salary_count = sum(1 for d in all_data
                       if d.get('salary_number', 0) > 0)
    log_progress(f"\n  실제 연봉 정보 보유: {salary_count}건 / {len(all_data)}건 "
          f"({salary_count/len(all_data)*100:.1f}%)")

    log_progress(f"\n  저장 경로:")
    log_progress(f"    - {output_path} (전체 데이터)")
    log_progress(f"    - {compat_path} (탭1/탭3 호환)")
    log_progress(f"\n  완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_progress("=" * 70)

    return all_data


if __name__ == "__main__":
    run_crawling()
