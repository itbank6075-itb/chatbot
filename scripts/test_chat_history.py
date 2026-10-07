"""사용자의 기록을 건드리지 않고 대화 저장/초기화를 검사합니다."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from test_rag_app import check_ui
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore

with TemporaryDirectory() as folder, patch.object(app, "HISTORY_DB", Path(folder) / "history.sqlite3"):
    user = {"role": "user", "content": "기존 대화"}
    assistant = {"role": "assistant", "answer": "저장된 답변", "sources": [
        {"file": "sample.pdf", "page": 2, "quote": "저장할 근거 문장입니다."}
    ]}
    assert app.load_history([user]) == [user]
    assert app.load_history([user]) == [user], "Migration must happen only once"
    app.save_message(assistant)
    assert app.load_history() == [user, assistant]
    # 다른 프로세스에서도 복원되는지 실제 새 Python 프로세스로 확인합니다.
    import subprocess
    code = "from pathlib import Path; import app; app.HISTORY_DB=Path(" + repr(str(app.HISTORY_DB)) + "); assert len(app.load_history()) == 2; assert app.load_history()[1]['sources'][0]['page'] == 2"
    subprocess.run([sys.executable, "-W", "error", "-c", code], cwd=app.ROOT, check=True)
    app.clear_history()
    assert app.load_history([user, assistant]) == [], "Reset must not resurrect stale session messages"
print("PASS: migration, durable storage, sources, process restart, reset")

doc = Document(page_content="국내 출장에는 일비와 식비, 숙박비가 지급됩니다.",
               metadata={"source": "fixture.pdf", "page": 3})
store = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
store.add_documents([doc])
index = app.Index(store, [p.name for p in app.data_files()], 154, 1, 0)
chain = RunnableLambda(lambda payload: app.GroundedAnswer(found=True, statements=[
    app.Statement(answer="국내 출장에는 일비, 식비, 숙박비가 지급됩니다.",
                  evidence=[app.Evidence(reference_id=1)])
]))
check_ui(index, chain, "offline-test-key")
print("PASS: browser refresh, index rebuild, reset button, cross-session reset")
