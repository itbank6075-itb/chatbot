"""검증기가 오답·가짜 인용·누락을 감지하는지 API 없이 검사합니다."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from langchain_core.documents import Document
from rag_validation import assess_response, run_validation, load_cases, FAIL, REVIEW, SKIP

text = "운임 지급에는 상한이 있습니다."
doc = Document(page_content=text, metadata={"source": "fixture.txt", "page": 1})
case = {"id": "known", "question": "known", "answerable": True, "reviewed": False,
        "expected_terms": ["상한"], "forbidden_terms": ["무제한"],
        "expected_sources": [{"file": "fixture.txt", "page": 1, "anchor": "상한"}]}
correct = {"answer": text, "sources": [{"file": "fixture.txt", "page": 1, "quote": text}]}
assert not assess_response(case, correct, [doc], app.NOT_FOUND)
for response in [
    {"answer": "무제한", "sources": correct["sources"]},
    {"answer": text, "sources": [{"file": "fixture.txt", "page": 1, "quote": "실제로 존재하지 않는 인용문"}]},
    {"answer": text, "sources": [{"file": "fixture.txt", "page": 99, "quote": text}]},
    {"answer": app.NOT_FOUND, "sources": []},
]:
    assert assess_response(case, response, [doc], app.NOT_FOUND)
unknown = {"id": "unknown", "question": "unknown", "answerable": False, "reviewed": False}
assert not assess_response(unknown, {"answer": app.NOT_FOUND, "sources": []}, [doc], app.NOT_FOUND)
assert assess_response(unknown, correct, [doc], app.NOT_FOUND)
print("PASS: wrong answer, hallucinated quotes, wrong page, false rejection, unknown questions")

with TemporaryDirectory() as directory:
    root = Path(directory)
    file = root / "fixture.txt"
    file.write_text(text, encoding="utf-8")
    config_path = root / "cases.json"
    config = {"baseline_sha256": {}, "cases": [case, unknown]}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    runtime = SimpleNamespace(
        DATA_DIR=root, data_files=lambda: [file],
        read_documents=lambda files: ([doc], 1, 0), split_documents=lambda docs: docs,
        EMBEDDING_MODEL=app.EMBEDDING_MODEL, CHAT_MODEL=app.CHAT_MODEL,
        NOT_FOUND=app.NOT_FOUND, GroundedAnswer=app.GroundedAnswer,
        search_context=app.search_context, validate_answer=app.validate_answer,
    )
    report = run_validation(runtime, config_path)
    assert not report["counts"][FAIL] and report["counts"][REVIEW] and report["counts"][SKIP]
    runtime.split_documents = lambda docs: [Document(page_content="missing content", metadata=doc.metadata)]
    assert run_validation(runtime, config_path)["counts"][FAIL]
    runtime.split_documents = lambda docs: docs
    class Store:
        def similarity_search(self, question, k):
            return [doc]
    class Chain:
        def invoke(self, payload):
            if payload["question"] == "unknown":
                return app.GroundedAnswer(found=False, statements=[])
            return app.GroundedAnswer(found=True, statements=[app.Statement(answer=text, evidence=[app.Evidence(reference_id=1)])])
    report = run_validation(runtime, config_path, Store(), Chain())
    assert not report["counts"][FAIL] and len(report["cases"]) == 2
    class LeakyChain:
        def invoke(self, payload):
            raise RuntimeError("SECRET_DO_NOT_LOG")
    report = run_validation(runtime, config_path, Store(), LeakyChain())
    assert report["counts"][FAIL] and "SECRET_DO_NOT_LOG" not in json.dumps(report)
    config["cases"].append(case)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    try:
        load_cases(config_path)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate case IDs must be rejected")
print("PASS: split loss, review flags, live result grading, API secret protection, invalid config")
