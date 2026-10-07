"""사용자가 보고한 두 질문을 실제 OpenAI API로 재현합니다."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app

print("Building index for real reported questions...", flush=True)
key = app.read_api_key()
try:
    index = app.build_index(app.data_files(), key)
    chain = app.make_chain(key)
    for question in [
        "근무지 외 국내출장 시 지급되는 여비 항목은 무엇인가요?",
        "서울 거주·세종 근무자가 서울에서 대구로 바로 출장 가면 운임은 어떻게 지급되나요?",
    ]:
        docs = index.store.similarity_search(question, k=8)
        result = chain.invoke({"question": question, "context": app.search_context(docs)})
        response = app.validate_answer(result, docs)
        print(json.dumps({"question": question, **response}, ensure_ascii=True), flush=True)
        assert response["sources"], "A reported question must have verified supporting evidence"
        if "세종" in question:
            assert "대전" not in response["answer"], "Do not substitute the document example city for the user's workplace"
            assert "세종" in response["answer"], "Preserve the workplace specified in the question"

    absent = app.answer_question("제 개인 은행 계좌번호는 무엇인가요?", index.store, chain)
    assert not absent["sources"] and absent["answer"] == app.NOT_FOUND
    print("PASS: both reported questions answered with sources; unsupported question refused", flush=True)
except Exception as exc:
    print("FAILED:", type(exc).__name__, flush=True)
    if isinstance(exc, AssertionError):
        print(str(exc), flush=True)
    raise SystemExit(1) from None
