"""Streamlit이 파일 전체를 다시 실행할 때 자료형이 달라지는 문제를 검사합니다."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import importlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.app_test import TMP_DIR
import atexit
atexit.register(TMP_DIR.cleanup)

text = "근무지 외 국내출장 시에는 운임, 일비, 숙박비, 식비를 지급합니다."
doc = Document(page_content=text, metadata={"source": "fixture.pdf", "page": 1})
old_result = app.GroundedAnswer(found=True, statements=[
    app.Statement(answer=text, evidence=[app.Evidence(reference_id=1)])
])
importlib.reload(app)
assert not isinstance(old_result, app.GroundedAnswer)
assert app.validate_answer(old_result, [doc])["sources"], "Cached response must survive class redefinition"
assert not app.validate_answer({"invalid": True}, [doc])["sources"]
print("PASS: cached Pydantic response survives module reload")

class FakePage:
    def extract_text(self):
        return text

class FakeReader:
    is_encrypted = False
    pages = [FakePage()]
    def __init__(self, path):
        pass

class FakeChat:
    def __init__(self, **kwargs):
        pass
    def with_structured_output(self, schema, **kwargs):
        # 처음 생성된 자료형을 보관하는 실제 체인의 캐시 동작을 재현합니다.
        return RunnableLambda(lambda payload: schema.model_validate({
            "found": True, "statements": [{"answer": text, "evidence": [{"reference_id": 1}]}]
        }))

with TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "DATA").mkdir()
    (root / "DATA" / "fixture.pdf").write_bytes(b"offline fixture")
    (root / ".env").write_text("OPENAI_API_KEY=offline-test-key\n", encoding="utf-8")
    (root / "app.py").write_bytes(Path(app.__file__).read_bytes())
    import json
    from hashlib import sha256
    (root / "validation_cases.json").write_text(json.dumps({
        "baseline_sha256": {"fixture.pdf": sha256(b"offline fixture").hexdigest()},
        "cases": [{"id": "fixture", "question": "여비 항목", "answerable": True,
                   "expected_terms": ["일비"], "reviewed": False,
                   "expected_sources": [{"file": "fixture.pdf", "page": 1, "anchor": "일비"}]}]
    }, ensure_ascii=False), encoding="utf-8")

    with patch("pypdf.PdfReader", FakeReader), \
         patch("langchain_openai.OpenAIEmbeddings", lambda **kwargs: DeterministicFakeEmbedding(size=32)), \
         patch("langchain_openai.ChatOpenAI", FakeChat):
        # from_function은 기존 import를 재사용하므로 이 버그를 놓칩니다.
        # 실제 서버처럼 from_file로 app.py 전체를 재실행해야 합니다.
        at = AppTest.from_file(root / "app.py", default_timeout=60).run()
        assert not at.exception and not at.error
        for question in ["근무지 외 국내출장 시 지급되는 여비 항목은 무엇인가요?", "여비 항목을 다시 알려주세요."]:
            at.chat_input[0].set_value(question).run()
            assert not at.exception and not at.error
            answer = at.session_state["messages"][-1]
            assert answer["sources"], "Valid answers must survive Streamlit reruns"
            assert answer["answer"] == text
        assert len(at.session_state["messages"]) == 4
        before = list(at.session_state["messages"])
        at.sidebar.button[2].click().run()
        assert not at.exception and not at.error
        assert "validation_report" in at.session_state
        from rag_validation import FAIL
        assert not at.session_state["validation_report"]["counts"][FAIL]
        assert at.session_state["messages"] == before
        assert (root / ".chatbot" / "validation" / "latest.json").exists()

print("PASS: real app.py reruns retain valid answers and sources")

