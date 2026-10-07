"""PDF/RAG 검증: 자동으로 확인할 수 있는 사실과 사람의 검토를 구분합니다."""
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

PASS, FAIL, REVIEW, SKIP = "통과", "실패", "검토 필요", "미실행"


def compact(text):
    # PDF 줄바꿈과 띄어쓰기 차이만 무시합니다. 숫자와 글자는 바꾸지 않습니다.
    return re.sub(r"\s+", "", text)


def load_cases(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("검증 질문 목록(cases)이 비어 있습니다.")
    identifiers = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("각 검증 질문에는 문자열 id가 필요합니다.")
        if case["id"] in identifiers:
            raise ValueError("검증 질문 id가 중복되었습니다.")
        identifiers.add(case["id"])
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("검증 질문이 비어 있습니다.")
        if type(case.get("answerable")) is not bool:
            raise ValueError("answerable은 true 또는 false여야 합니다.")
        for key in ("expected_terms", "any_terms", "forbidden_terms"):
            terms = case.get(key, [])
            if not isinstance(terms, list) or any(not isinstance(t, str) or not t.strip() for t in terms):
                raise ValueError(f"{key}에는 비어 있지 않은 문자열 목록을 넣어주세요.")
        sources = case.get("expected_sources", [])
        if case["answerable"] and not sources:
            raise ValueError("답변 가능한 질문에는 expected_sources가 필요합니다.")
        for source in sources:
            if not isinstance(source.get("file"), str) or type(source.get("page")) is not int or source["page"] < 1:
                raise ValueError("출처에는 파일명과 1 이상의 페이지 번호가 필요합니다.")
            if not isinstance(source.get("anchor"), str) or not source["anchor"].strip():
                raise ValueError("출처 원문을 식별할 anchor가 필요합니다.")
    return data


def source_matches(source, doc):
    return (doc.metadata.get("source") == source["file"]
            and doc.metadata.get("page") == source["page"]
            and compact(source["anchor"]) in compact(doc.page_content))


def citation_errors(response, retrieved):
    errors = []
    for source in response.get("sources", []):
        # 존재하는 파일명만 검사하지 않고, 해당 페이지의 해당 원문인지 검사합니다.
        quote = compact(source.get("quote", ""))
        if not quote or not any(
            doc.metadata.get("source") == source.get("file")
            and doc.metadata.get("page") == source.get("page")
            and quote in compact(doc.page_content) for doc in retrieved
        ):
            errors.append("검색된 원문과 일치하지 않는 인용문")
    return errors


def assess_response(case, response, retrieved, not_found):
    errors = citation_errors(response, retrieved)
    answer = response.get("answer", "")
    sources = response.get("sources", [])
    if not case["answerable"]:
        if sources or answer != not_found:
            errors.append("문서 밖 질문을 거절하지 않음")
    else:
        if answer == not_found or not sources:
            errors.append("답변 가능한 질문을 거절했거나 출처가 없음")
        for term in case.get("expected_terms", []):
            if compact(term) not in compact(answer):
                errors.append("필수 내용 누락: " + term)
        if case.get("any_terms") and not any(compact(t) in compact(answer) for t in case["any_terms"]):
            errors.append("필수 지급 조건 또는 제한 설명 누락")
        for term in case.get("forbidden_terms", []):
            if compact(term) in compact(answer):
                errors.append("금지된 조건/내용 포함: " + term)
        # 정답 페이지를 검색했더라도 엉뚱한 페이지를 인용하면 통과시키지 않습니다.
        if not any(source_matches(expected, doc)
                   and any(s["file"] == doc.metadata["source"] and s["page"] == doc.metadata["page"]
                           and compact(expected["anchor"]) in compact(s["quote"])
                           for s in sources)
                   for expected in case["expected_sources"] for doc in retrieved):
            errors.append("기준 출처와 기준 원문을 답변에서 인용하지 않음")
    return errors


def run_validation(runtime, cases_path, store=None, chain=None):
    """store/chain을 주면 실제 API 검사, 없으면 PDF와 조각만 검사합니다."""
    config = load_cases(cases_path)
    checks, file_stats, results = [], [], []
    def check(stage, item, status, detail):
        checks.append({"단계": stage, "항목": item, "결과": status, "설명": detail})
    files = runtime.data_files()
    actual_hashes = {p.relative_to(runtime.DATA_DIR).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files}
    check("문서 버전", "기준 PDF 변경 여부", PASS if actual_hashes == config.get("baseline_sha256") else REVIEW,
          "검증 질문을 만든 당시 파일과 비교합니다. 변경된 문서는 기준 답안도 다시 검토해야 합니다.")
    try:
        documents, pages, empty_pages = runtime.read_documents(files)
    except (ValueError, OSError) as exc:
        check("추출", "모든 파일 읽기", FAIL, str(exc))
        return finish(runtime, checks, [], [], actual_hashes, config)
    grouped = {}
    for doc in documents:
        grouped.setdefault(doc.metadata["source"], []).append(doc)
    check("추출", "모든 파일 읽기", PASS if len(grouped) == len(files) else FAIL,
          f"파일 {len(files)}개, 페이지 {pages}개, 텍스트 없는 페이지 {empty_pages}개")
    if empty_pages:
        check("추출", "빈 페이지", REVIEW, "원본을 확인해 빈 페이지인지 OCR이 필요한 이미지인지 구분해주세요.")
    for filename, docs in grouped.items():
        text = "\n".join(d.page_content for d in docs)
        bad = text.count("\ufffd") + len(re.findall(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", text))
        suspicious_pages = [d.metadata["page"] for d in docs
                            if "\ufffd" in d.page_content or re.search(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", d.page_content)]
        file_stats.append({"파일": filename, "텍스트 페이지": len(docs), "글자 수": len(text),
                           "의심 문자": bad, "의심 문자 페이지": suspicious_pages})
        check("추출", filename + " 문자 검사", REVIEW if bad else PASS,
              f"의심 문자 {bad}개, 확인할 PDF 페이지: {suspicious_pages}. 문자 검사는 표의 배치나 금액 정확성까지 판별하지 않습니다.")
    chunks = runtime.split_documents(documents)
    for doc in documents:
        page_chunks = [c for c in chunks if c.metadata == doc.metadata]
        # 각 조각이 원문의 일부인지, 공백 이외의 글자가 빠졌는지 확인합니다.
        original = compact(doc.page_content)
        covered = set()
        for chunk in page_chunks:
            text = compact(chunk.page_content)
            start = 0
            while text and (position := original.find(text, start)) >= 0:
                covered.update(range(position, position + len(text)))
                start = position + 1
        good = bool(page_chunks) and len(covered) == len(original) and all(len(c.page_content) <= 1000 for c in page_chunks)
        if not good:
            check("분할", f"{doc.metadata['source']} · {doc.metadata['page']}페이지", FAIL,
                  "문서 조각의 내용 보존 또는 최대 길이 검사 실패")
    if not any(c["단계"] == "분할" and c["결과"] == FAIL for c in checks):
        check("분할", "내용 보존·조각 길이", PASS, f"{len(chunks)}개 조각, 공백 제외 원문 보존, 최대 1,000자")
    for case in config["cases"]:
        for source in case.get("expected_sources", []):
            exists = any(source_matches(source, doc) for doc in documents)
            check("기준 근거", case["id"] + " · " + source["file"], PASS if exists else FAIL,
                  f"PDF {source['page']}페이지에서 기준 원문(anchor)을 확인합니다.")
    check("원본 대조", "표·금액·각주·예외 조건", REVIEW,
          "자동으로 판정할 수 없습니다. PDF 원본과 추출 텍스트를 직접 대조해야 합니다.")
    check("기준 답안", "대표 질문의 사람 검토", PASS if all(c.get("reviewed") is True for c in config["cases"]) else REVIEW,
          "validation_cases.json의 reviewed는 기준 답안과 원본을 검토한 뒤 true로 바꿔주세요.")
    if store is None or chain is None:
        check("실제 검색·답변", "OpenAI API 검사", SKIP, "화면의 실제 검색·답변 검사 또는 CLI --live 옵션으로 실행합니다.")
    else:
        for case in config["cases"]:
            try:
                retrieved = store.similarity_search(case["question"], k=8)
                hit = not case["answerable"] or any(source_matches(s, doc) for s in case["expected_sources"] for doc in retrieved)
                if case["answerable"]:
                    check("검색", case["id"], PASS if hit else FAIL, "검색 결과 8개 안에 기준 페이지와 원문이 포함되는지 확인합니다.")
                raw = chain.invoke({"question": case["question"], "context": runtime.search_context(retrieved)})
                response = runtime.validate_answer(raw, retrieved)
                # 검증 형식 오류가 정상 거절로 위장해 통과하지 않도록 원 응답도 확인합니다.
                payload = raw.model_dump() if hasattr(raw, "model_dump") else raw
                parsed = runtime.GroundedAnswer.model_validate(payload)
                errors = assess_response(case, response, retrieved, runtime.NOT_FOUND)
                if not case["answerable"] and (parsed.found or parsed.statements):
                    errors.append("모델의 원 응답이 근거 없음 판정을 하지 않음")
                status = FAIL if errors else PASS
                check("답변" if case["answerable"] else "거절", case["id"], status,
                      "; ".join(errors) if errors else "필수 내용·질문 조건·기준 출처·원문 인용 검사 통과")
                results.append({"id": case["id"], "question": case["question"], "status": status,
                                "errors": errors, "response": response,
                                "retrieved_sources": [{"file": d.metadata["source"], "page": d.metadata["page"]} for d in retrieved]})
            except Exception as exc:
                # 인증 예외 원문에는 키가 포함될 수 있어 종류만 보고서에 남깁니다.
                check("API 실행", case["id"], FAIL, f"{type(exc).__name__} · {getattr(exc, 'code', None)} · HTTP {getattr(exc, 'status_code', None)}")
                results.append({"id": case["id"], "status": FAIL, "error_type": type(exc).__name__})
    return finish(runtime, checks, file_stats, results, actual_hashes, config)


def finish(runtime, checks, file_stats, results, hashes, config):
    counts = Counter(item["결과"] for item in checks)
    return {"created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
            "overall": FAIL if counts[FAIL] else REVIEW if counts[REVIEW] else PASS,
            "mode": "live" if results else "offline", "models": {"embedding": runtime.EMBEDDING_MODEL, "chat": runtime.CHAT_MODEL},
            "counts": {status: counts[status] for status in (PASS, FAIL, REVIEW, SKIP)},
            "file_sha256": hashes, "case_count": len(config["cases"]), "checks": checks,
            "files": file_stats, "cases": results,
            "note": "통과는 지정된 자동 검사 통과를 뜻합니다. 모든 답변의 의미적 정확성이나 전체 PDF 원본 일치를 보장하지 않습니다."}


def save_report(report, directory, filename="latest.json"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    temporary = directory / (filename + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def failure_report(runtime, error_type, error_code=None, http_status=None):
    # API 요청이 시작되지 못한 경우도 성공 보고서로 오해하지 않도록 별도로 저장합니다.
    return {"created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
            "overall": FAIL, "mode": "live", "counts": {PASS: 0, FAIL: 1, REVIEW: 0, SKIP: 0},
            "checks": [{"단계": "API 실행", "항목": "검증 실행 준비", "결과": FAIL,
                        "설명": f"{error_type} · {error_code} · HTTP {http_status}"}],
            "api_error": {"type": error_type, "code": error_code, "http_status": http_status},
            "models": {"embedding": runtime.EMBEDDING_MODEL, "chat": runtime.CHAT_MODEL},
            "cases": [], "note": "실제 검색·답변 검사가 실행되지 못했습니다. 이 보고서는 답변 정확도 검사 결과가 아닙니다."}
