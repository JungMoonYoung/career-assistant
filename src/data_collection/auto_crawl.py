"""
주간 자동 크롤링 스크립트
- 매주 1회 실행, 약 500건 크롤링
- 종료 없이 매주 계속 실행
- 기존 데이터와 병합 + 중복 제거
- 벡터 DB 자동 재빌드
"""
import sys
import os
import json
import time
import random
import logging
from datetime import datetime

# 프로젝트 루트를 작업 디렉토리로 설정 (Task Scheduler 대응)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

STATE_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "_auto_crawl_state.json")
LOG_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "_auto_crawl_weekly.log")
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "crawled_jobs_all.json")
COMPAT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "saramin_10_jobs.json")

# 주간 목표: ~500건 (불균등 분포)
ROLE_TARGETS_WEEKLY = {
    "백엔드 개발자":     65,
    "데이터 분석가":     60,
    "프론트엔드 개발자": 55,
    "데이터 엔지니어":   50,
    "일반사무":          45,
    "UI/UX 디자이너":   40,
    "퍼포먼스 마케터":   40,
    "인사 총무":         35,
    "경리 회계":         35,
    "영업 지원":         40,
    "고객상담 CS":       35,
}


def setup_logging():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_dedup_key(job):
    url = job.get('url', '')
    if url:
        return url
    source = job.get('source', '')
    company = job.get('company', {}).get('detail', {}).get('name', '')
    title = job.get('position', {}).get('title', '')
    return f"{source}|{company}|{title}"


def merge_and_deduplicate(new_jobs, existing_path):
    if os.path.exists(existing_path):
        with open(existing_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    else:
        existing = []

    seen = set()
    for job in existing:
        seen.add(get_dedup_key(job))

    added = 0
    for job in new_jobs:
        key = get_dedup_key(job)
        if key not in seen:
            existing.append(job)
            seen.add(key)
            added += 1

    random.shuffle(existing)

    # 임시 파일에 먼저 쓴 후 교체 (안전)
    tmp_path = existing_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, existing_path)

    return existing, added


def crawl_weekly():
    """~500건 크롤링"""
    from src.data_collection.crawl_all import SaraminCrawler, JobkoreaCrawler, log_progress

    saramin = SaraminCrawler()
    jobkorea = JobkoreaCrawler()
    all_jobs = []

    roles = list(ROLE_TARGETS_WEEKLY.items())
    random.shuffle(roles)

    for idx, (role, target) in enumerate(roles, 1):
        saramin_target = int(target * 0.6)
        jobkorea_target = target - saramin_target

        logging.info(f"[{idx}/{len(roles)}] {role} (사람인 {saramin_target} + 잡코리아 {jobkorea_target})")

        s_jobs = saramin.crawl_role(role, saramin_target)
        logging.info(f"  사람인: {len(s_jobs)}건")
        all_jobs.extend(s_jobs)

        time.sleep(random.uniform(3.0, 6.0))

        j_jobs = jobkorea.crawl_role(role, jobkorea_target)
        logging.info(f"  잡코리아: {len(j_jobs)}건")
        all_jobs.extend(j_jobs)

        if idx < len(roles):
            time.sleep(random.uniform(5.0, 10.0))

    total_requests = saramin.request_count + jobkorea.request_count
    logging.info(f"크롤링 완료: {len(all_jobs)}건, 총 요청 {total_requests}회")
    return all_jobs


def rebuild_vector_db():
    """벡터 DB 재빌드 — 크롤링 후 자동으로 새 데이터를 벡터 DB에 반영"""
    try:
        # update_vector_db.py는 PROJECT_ROOT에 위치
        sys.path.insert(0, PROJECT_ROOT)
        from update_vector_db import update_vector_database
        logging.info("벡터 DB 재빌드 시작...")
        success = update_vector_database(json_path=DATA_FILE, batch_size=500, silent=True)
        if success:
            logging.info("벡터 DB 재빌드 완료 — 새 크롤링 데이터가 벡터 DB에 반영됨")
        else:
            logging.warning("벡터 DB 재빌드 실패")
        return success
    except Exception as e:
        logging.error(f"벡터 DB 재빌드 오류: {e}")
        return False


def main():
    setup_logging()
    logging.info("=" * 60)
    logging.info("주간 자동 크롤링 시작")
    logging.info("=" * 60)

    # 상태 확인
    state = load_state()
    if state is None:
        state = {
            'first_run': datetime.now().isoformat(),
            'run_count': 0,
            'history': []
        }
        logging.info("첫 실행 - 상태 파일 생성")

    state['run_count'] += 1
    run_num = state['run_count']
    save_state(state)
    logging.info(f"실행 #{run_num}")

    # 크롤링
    start = time.time()
    new_jobs = crawl_weekly()

    # 병합
    merged, added = merge_and_deduplicate(new_jobs, DATA_FILE)

    # 호환 파일도 업데이트
    tmp_compat = COMPAT_FILE + '.tmp'
    with open(tmp_compat, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    os.replace(tmp_compat, COMPAT_FILE)

    elapsed = (time.time() - start) / 60
    logging.info(f"크롤링 {len(new_jobs)}건 중 신규 {added}건 추가, 전체 {len(merged)}건")
    logging.info(f"소요 시간: {elapsed:.1f}분")

    # 벡터 DB 재빌드
    rebuild_vector_db()

    # 실행 기록 저장
    state['history'].append({
        'run': run_num,
        'date': datetime.now().isoformat(),
        'crawled': len(new_jobs),
        'new_added': added,
        'total': len(merged),
        'elapsed_min': round(elapsed, 1),
    })
    save_state(state)

    logging.info(f"다음 실행은 1주 후입니다.")
    logging.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"예상치 못한 오류: {e}", exc_info=True)
        sys.exit(1)
