"""PDF 검증 CLI: 기본은 API 없는 검사, --live는 실제 검색과 답변까지 검사합니다."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from rag_validation import run_validation, save_report, failure_report, FAIL

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--live", action="store_true", help="OpenAI API로 대표 질문의 검색/답변을 검사")
parser.add_argument("--cases", type=Path, default=app.ROOT / "validation_cases.json")
parser.add_argument("--output", type=Path, default=app.ROOT / ".chatbot" / "validation")
args = parser.parse_args()
try:
    store = chain = None
    if args.live:
        key = app.read_api_key()
        if not key:
            print("OPENAI_API_KEY is not configured.")
            raise SystemExit(2)
        print("Building embeddings for live validation...", flush=True)
        store = app.build_index(app.data_files(), key).store
        chain = app.make_chain(key)
    report = run_validation(app, args.cases, store, chain)
    path = save_report(report, args.output)
    # Windows 터미널 출력 인코딩과 관계없이 집계가 읽히도록 숫자를 출력합니다.
    print("RESULT_COUNTS:", list(report["counts"].values()), "(pass, fail, review, skipped)")
    print("REPORT:", path)
    for case in report["cases"]:
        print("CASE:", case["id"], "FAIL" if case["status"] == FAIL else "PASS")
    raise SystemExit(1 if report["counts"][FAIL] else 0)
except SystemExit:
    raise
except Exception as exc:
    report = failure_report(app, type(exc).__name__, getattr(exc, "code", None), getattr(exc, "status_code", None))
    path = save_report(report, args.output, "api_failure.json")
    print("VALIDATION_FAILED:", type(exc).__name__)
    print("ERROR_CODE:", getattr(exc, "code", None))
    print("ERROR_REPORT:", path)
    raise SystemExit(2) from None
