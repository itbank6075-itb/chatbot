# RAG chatbot development environment

Python 3.11; dependencies are recorded in pyproject.toml and uv.lock.
Pre-release packages are disabled. Keep uv.lock for reproducible installs.

## PowerShell quick start

uv is currently installed outside PATH. Use this session-local alias:

```powershell
Set-Alias uv "$env:USERPROFILE\.local\bin\uv.exe"
uv sync --locked
.\.venv\Scripts\python.exe -S scripts\repair_venv_path.py
uv run --locked python -W error scripts\check_environment.py
uv pip check
```

If uv itself fails with UnicodeDecodeError before synchronization, run the
repair command first, then synchronize, then repair again. Python 3.11 on
Windows reads editable .pth files using the system encoding; uv_build writes
this project's Korean path in UTF-8. The repair writes the current source
path using ASCII escapes, retaining editable source imports. Repeat the
repair after reinstalling the local project or moving the folder.

## Application development

Use current imports:

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
```

For local development, copy .env.example to .env and provide OPENAI_API_KEY. In Streamlit Cloud, register OPENAI_API_KEY in app Secrets.
The environment check runs offline without reading or printing API keys.
Create app.py when ready, then start Streamlit with:

```powershell
uv run --locked streamlit run app.py
```

The RAG application is implemented in app.py; see the Korean quick start below.

References:
- https://reference.langchain.com/python/langchain-openai/langchain_openai
- https://docs.streamlit.io/get-started/installation/command-line

## RAG 챗봇 실행

프로젝트 최상단의 `app.py`가 실제 RAG 챗봇입니다.

```powershell
Set-Alias uv "$env:USERPROFILE\.local\bin\uv.exe"
uv run --locked streamlit run app.py
```

- Streamlit Secrets의 `OPENAI_API_KEY`를 우선 읽고, 설정이 없으면 로컬 `.env`를 읽습니다. 키를 화면이나 로그에 출력하지 않습니다.
- `DATA`의 모든 하위 폴더를 읽으며 PDF, TXT, MD, CSV, JSON을 지원합니다.
- 임베딩: `text-embedding-3-small`, 답변: `gpt-4o-mini`.
- 벡터DB는 `InMemoryVectorStore`입니다. 재시작하거나 문서를 다시 읽으면 임베딩을 다시 생성합니다.
- 문서 내용이 변경되면 다음 화면 실행 시 인덱스를 갱신합니다.
- 문서에 없는 질문은 근거를 찾을 수 없다고 답합니다. 모든 답변 항목에 원문 근거를 요구하고 실제 문서에 있는 인용문인지 검사합니다.
- 출처는 DATA 기준 파일 경로와 PDF 페이지 번호(첫 페이지가 1)로 표시합니다. PDF 본문에 인쇄된 쪽수와 다를 수 있습니다.
- 대화 내용은 화면에 보관하지만 검색은 매 질문을 기준으로 합니다. 대상과 조건을 질문에 명시해주세요.
- 스캔 PDF의 이미지는 OCR 대상이며 텍스트가 없는 페이지는 안내합니다. 읽을 수 없는 파일이 있으면 일부 파일을 몰래 제외하지 않고 오류를 표시합니다.

검사 명령:

```powershell
uv run --locked python -W error scripts\test_rag_app.py
# 실제 OpenAI 호출을 포함하며 API 사용량이 발생합니다.
uv run --locked python -W error scripts\test_rag_app.py --live
```



## 대화 자동 저장과 초기화

질문, 답변, 출처는 `.chatbot/history.sqlite3`에 자동 저장합니다.
새로고침, 서버 재시작, 문서 재읽기 후에도 대화가 유지됩니다.
사이드바의 **대화 초기화** 버튼을 누르면 저장된 대화가 삭제됩니다.
이 로컬 앱의 대화 기록은 같은 프로젝트에 접속한 브라우저가 공유합니다.
대화 기록 파일은 Git에 포함하지 않습니다.

## 데이터 검증

사이드바의 **데이터 검증 실행**은 PDF 추출, 문서 조각 보존, 기준 원문의 존재 여부를 검사합니다.
**실제 검색·답변 검사**를 선택하면 현재 메모리 벡터DB를 재사용해 대표 질문의 검색과 답변도 검사합니다.
검증 결과는 화면에서 볼 수 있고 JSON 보고서를 다운로드할 수 있습니다. 대화 기록은 변경하지 않습니다.

```powershell
uv run --locked python -W error scripts\validate_data.py
uv run --locked python -W error scripts\validate_data.py --live
```

CLI의 --live는 새 메모리 벡터DB를 만들기 때문에 임베딩/답변 API 사용량이 발생합니다.
기준 질문은 `validation_cases.json`, 최신 보고서는 `.chatbot/validation/latest.json`에 저장합니다.
질문별 expected_terms는 필수 내용, any_terms는 최소 하나 필요한 표현,
forbidden_terms는 잘못 들어가면 실패할 조건, expected_sources는 기준 출처/페이지/원문입니다.
출처가 여러 개면 그중 하나를 검색하고 인용해야 합니다.

현재 대표 질문은 4개입니다. reviewed는 원본과 기준 답안을 사람이 확인한 뒤 true로 바꾸세요.
자동 검사가 통과해도 전체 PDF의 표/금액/각주 대조와 답변의 의미적 정확성은 별도 검토가 필요합니다.
문서 파일 해시가 바뀌면 기준 질문의 재검토가 필요하다고 표시합니다.
검사 실패 시 CLI 종료 코드는 1, 설정/실행 오류는 2입니다. 검토 필요 항목은 실패와 구분합니다.
"# chatbot" 


## Streamlit Community Cloud 배포와 API 키

1. 변경된 코드와 .gitignore를 GitHub main에 반영합니다. 실제 키는 커밋하지 않습니다.
2. https://share.streamlit.io 에서 Create app을 선택합니다.
3. Repository: `itbank6075-itb/chatbot`, Branch: `main`, Main file path: `app.py`.
4. Advanced settings에서 Python 3.11을 선택합니다.
5. 같은 화면의 Secrets 입력란에 다음 TOML을 넣습니다. 따옴표 안만 실제 키로 교체하세요.

```toml
OPENAI_API_KEY = "여기에 실제 OpenAI API 키 입력"
```

6. Save 후 Deploy합니다. 이미 배포한 앱은 앱 Settings의 Secrets에서 변경할 수 있습니다.

로컬 .env와 .streamlit/secrets.toml은 Git에서 제외합니다. 실제 Secrets 파일은 이 프로젝트에 생성하지 않습니다.
공개해도 되는 .env.example은 빈 키 항목만 포함합니다.
기존 uv.lock과 pyproject.toml을 사용하므로 별도의 requirements.txt를 추가하지 않았습니다.
Cloud에서는 uv.lock을 우선 인식합니다.

현재 대화 기록은 프로젝트 전체 접속자가 공유하는 로컬 SQLite 구조입니다.
개인별 대화가 필요한 공개 서비스로 운영하려면 사용자별 기록 분리가 필요합니다.
SQLite 기록은 Cloud 로컬 파일이므로 재배포/컨테이너 교체 후 영구 보존을 보장하지 않습니다.

공식 안내:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
