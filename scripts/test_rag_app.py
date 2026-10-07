"""RAG의 근거 검사와 실제 Streamlit 화면을 테스트합니다.

--live 옵션은 .env의 키로 실제 OpenAI 임베딩/답변 API를 호출합니다.
"""
from pathlib import Path
import json
import sys
from unittest.mock import patch
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("error")
import app
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.app_test import TMP_DIR
import atexit
atexit.register(TMP_DIR.cleanup)


def ui_entry():
    import app
    app.main()


def check_validation():
    doc = Document(page_content="국내 출장에는 일비와 식비, 숙박비가 지급됩니다.",
                   metadata={"source": "fixture.pdf", "page": 3})
    evidence = app.Evidence(reference_id=1)
    valid = app.GroundedAnswer(found=True, statements=[
        app.Statement(answer="국내 출장 지급 항목입니다.", evidence=[evidence])
    ])
    assert app.validate_answer(valid, [doc])["sources"][0]["page"] == 3
    invalid = valid.model_copy(deep=True)
    invalid.statements[0].evidence[0].reference_id = 999
    assert app.validate_answer(invalid, [doc])["answer"] == app.NOT_FOUND
    invalid.statements[0].evidence[0].reference_id = 0
    assert not app.validate_answer(invalid, [doc])["sources"]
    no_evidence = app.GroundedAnswer(found=True, statements=[
        app.Statement(answer="근거 없는 답변", evidence=[])
    ])
    assert app.validate_answer(no_evidence, [doc])["answer"] == app.NOT_FOUND
    absent = app.GroundedAnswer(found=False, statements=[])
    assert app.validate_answer(absent, [doc])["answer"] == app.NOT_FOUND
    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
    store.add_documents([doc])
    chain = RunnableLambda(lambda payload: valid)
    assert app.answer_question("국내 출장 지급 항목", store, chain)["sources"]
    empty = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
    assert app.answer_question("질문", empty, chain)["answer"] == app.NOT_FOUND
    # 하위 폴더, 대문자 확장자, 지원하지 않는 파일, 스캔 문서 오류도 검사합니다.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as folder:
        root = Path(folder)
        nested = root / "nested"
        nested.mkdir()
        (nested / "sample.TXT").write_text(doc.page_content, encoding="utf-8")
        with patch.object(app, "DATA_DIR", root):
            files = app.data_files()
            docs, pages, _ = app.read_documents(files)
            assert len(docs) == pages == 1
            assert docs[0].metadata["source"] == "nested/sample.TXT"
            before = app.fingerprint(files)
            files[0].write_text("새로운 문서 내용입니다.", encoding="utf-8")
            assert app.fingerprint(files) != before
            (root / "unsupported.bin").write_bytes(b"test")
            try:
                app.data_files()
            except ValueError:
                pass
            else:
                raise AssertionError("Unsupported files must not be silently skipped")
    print("PASS: citations, unknown answers, retrieval, nested files, file updates")


def check_ui(index, chain, key):
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory, patch.object(app, "HISTORY_DB", Path(directory) / "history.sqlite3"):
        _check_ui(index, chain, key)


def _check_ui(index, chain, key):
    # 입력 폼, 답변과 출처 표시, 대화 지우기를 실제 Streamlit 런타임에서 검사합니다.
    with patch.object(app, "read_api_key", return_value=key), \
         patch.object(app, "build_index", return_value=index), \
         patch.object(app, "make_chain", return_value=chain):
        at = AppTest.from_function(ui_entry, default_timeout=180)
        at.run()
        assert not at.exception, "Streamlit startup exception"
        assert not at.error, "Streamlit startup error"
        assert len(at.chat_input) == 1
        at.chat_input[0].set_value("여비의 종류에는 무엇이 있나요?").run()
        assert not at.exception and not at.error, "Streamlit chat error"
        messages = at.session_state["messages"]
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["sources"], "Answer must include source quotes"
        assert any(c.value == "출처 및 근거 문장" or "출처 및 근거 문장" in c.value
                   for c in at.markdown)
        saved = list(messages)
        # 문서를 다시 읽어도 대화는 유지되어야 합니다.
        at.sidebar.button[0].click().run()
        assert at.session_state["messages"] == saved
        # 새 브라우저 세션에 해당하는 AppTest에서도 저장된 대화를 복원합니다.
        fresh = AppTest.from_function(ui_entry, default_timeout=180).run()
        assert not fresh.exception and not fresh.error
        assert fresh.session_state["messages"] == saved
        assert fresh.sidebar.button[1].label == "대화 초기화"
        fresh.sidebar.button[1].click().run()
        assert not fresh.session_state["messages"]
        at.run()
        assert not at.session_state["messages"]
        assert app.load_history() == []
    with patch.object(app, "read_api_key", return_value=""):
        at = AppTest.from_function(ui_entry, default_timeout=30).run()
        assert not at.exception and at.info
        assert not at.chat_input
    print("PASS: Streamlit startup, question input, sources, clear chat, missing key")


def main():
    check_validation()
    files = app.data_files()
    docs, pages, empty_pages = app.read_documents(files)
    assert len({d.metadata["source"] for d in docs}) == len(files)
    print(f"PASS: all {len(files)} DATA files read; {pages} pages; {empty_pages} empty pages")
    if "--live" in sys.argv:
        key = app.read_api_key()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        try:
            print("LIVE: creating embeddings", flush=True)
            index = app.build_index(files, key)
            chain = app.make_chain(key)
            retrieved = index.store.similarity_search("여비의 종류에는 무엇이 있나요?", k=8)
            raw = chain.invoke({"question": "여비의 종류에는 무엇이 있나요?",
                                "context": app.search_context(retrieved)})
            answer = app.validate_answer(raw, retrieved)
            if not answer["sources"]:
                print("GROUNDING_DIAGNOSTIC:", raw.model_dump_json(), flush=True)
            assert answer["sources"], "Live answer did not contain verified evidence"
            print(f"PASS: live OpenAI embeddings ({index.chunks} chunks) and grounded answer")
            absent = app.answer_question(
                "이 문서에 적힌 제 개인 은행 계좌번호와 제 반려동물의 이름은 무엇인가요?",
                index.store, chain,
            )
            assert absent["answer"] == app.NOT_FOUND and not absent["sources"]
            print("PASS: live out-of-document question refused")
            check_ui(index, chain, key)
        except Exception as exc:
            # 인증 실패 때도 API 키를 로그로 출력하지 않습니다.
            print("LIVE_CHECK_FAILED:", type(exc).__name__)
            if isinstance(exc, AssertionError):
                print(str(exc))
            raise SystemExit(1) from None
    else:
        doc = Document(page_content="국내 출장에는 일비와 식비, 숙박비가 지급됩니다.",
                       metadata={"source": "fixture.pdf", "page": 3})
        store = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
        store.add_documents([doc])
        index = app.Index(store, [p.name for p in files], pages, 1, empty_pages)
        chain = RunnableLambda(lambda payload: app.GroundedAnswer(found=True, statements=[
            app.Statement(answer="국내 출장에는 일비, 식비, 숙박비가 지급됩니다.",
                          evidence=[app.Evidence(reference_id=1)])
        ]))
        check_ui(index, chain, "offline-test-key")
    print("ALL RAG CHECKS PASSED (warnings treated as errors)")


if __name__ == "__main__":
    main()

