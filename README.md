# 공무원 여비 RAG 챗봇

`DATA` 폴더의 공무원 여비 문서를 검색하고, 검색된 근거를 바탕으로 답변하는 Streamlit 챗봇입니다. 답변 아래에 출처 파일명, PDF 페이지 번호, 근거 문장을 표시합니다.

## 주요 기능

- PDF, TXT, MD, CSV, JSON 파일을 하위 폴더까지 읽습니다.
- 문서에서 근거를 찾지 못하면 답변할 수 없다고 안내합니다.
- 질문과 답변, 출처를 자동 저장하고 **대화 초기화** 버튼으로 삭제할 수 있습니다.
- **문서 다시 읽기** 버튼으로 변경된 문서를 다시 반영할 수 있습니다.
- 화면과 CLI에서 문서 추출 및 대표 질문의 검색·답변을 검증할 수 있습니다.
- 로컬에서는 `.env`, Streamlit Cloud에서는 Secrets로 API 키를 설정합니다.

## 기술 구성

| 항목 | 사용 기술 |
| --- | --- |
| Python | 3.11 |
| 패키지 관리 | uv, `pyproject.toml`, `uv.lock` |
| 화면 | Streamlit |
| RAG 구성 | LangChain 프롬프트·Runnable 연결 및 `invoke` |
| 임베딩 모델 | OpenAI `text-embedding-3-small` |
| 답변 모델 | OpenAI `gpt-4o-mini` |
| 벡터 저장소 | LangChain `InMemoryVectorStore` |
| PDF 텍스트 추출 | pypdf |
| 대화 저장 | SQLite |

`LLMChain`, `ConversationChain`, `RetrievalQA`는 사용하지 않습니다. 설치 버전은 `uv.lock`으로 고정하며 사전 출시 버전은 허용하지 않습니다.

## 로컬 실행

Python 3.11과 uv가 준비된 환경에서 프로젝트 최상위 폴더에서 실행합니다.

### 1. 의존성 설치

```powershell
uv sync --locked
```

Windows에서 uv 명령을 찾지 못하고 사용자 기본 경로에 설치되어 있다면, 다음 명령으로 현재 터미널에 별칭을 설정합니다.

```powershell
Set-Alias uv "$env:USERPROFILE\.local\bin\uv.exe"
```

### 2. API 키 설정

`.env.example`을 복사합니다. 이미 `.env`가 있다면 복사하지 말고 기존 파일을 수정하세요.

```powershell
Copy-Item .env.example .env
```

`.env` 파일의 빈 값에 본인의 API 키를 입력합니다.

```dotenv
OPENAI_API_KEY=여기에_실제_API_키_입력
```

실제 키를 코드나 README에 작성하지 마세요. `.env`, `.env` 백업 파일, `.venv`, `.streamlit/secrets.toml`은 `.gitignore`에 등록되어 있습니다. `.env.example`에는 빈 설정 항목만 저장합니다.

키는 Streamlit Secrets의 `OPENAI_API_KEY`를 우선 읽으며, 해당 항목이 없으면 프로젝트의 `.env`를 읽습니다. Secrets에 빈 문자열을 넣으면 키가 없는 상태로 처리됩니다. 잘못된 TOML 설정은 오류로 안내합니다.

### 3. 문서 준비 및 실행

프로젝트 최상위의 `DATA` 폴더에 문서를 넣은 뒤 실행합니다.

```powershell
uv run --locked streamlit run app.py
```

브라우저에서 [http://localhost:8501](http://localhost:8501)을 엽니다. 최초 문서 임베딩과 질문 답변에는 OpenAI API 사용량이 발생합니다.

## 문서 검색과 답변 방식

1. PDF를 페이지별로 추출하고 출처 파일명과 페이지 번호를 기록합니다.
2. `RecursiveCharacterTextSplitter`로 최대 1,000자, 겹침 200자 기준으로 문서를 나눕니다.
3. 문서 조각을 `text-embedding-3-small`로 임베딩하고 메모리 벡터 저장소에 넣습니다.
4. 질문과 관련된 문서 조각을 검색해 `gpt-4o-mini`에 전달합니다.
5. 구조화된 답변의 근거 참조를 확인하고 답변과 출처를 표시합니다.

벡터 저장소는 메모리에 있으므로 서버를 재시작하면 임베딩을 다시 생성합니다. 문서 내용 변경도 재색인 대상입니다. PDF 페이지 번호는 파일의 첫 페이지를 1로 세며 문서에 인쇄된 쪽수와 다를 수 있습니다.

스캔 PDF의 OCR과 표 구조 복원은 구현되어 있지 않습니다. 빈 페이지는 안내하며, 텍스트가 전혀 없는 파일이나 지원하지 않는 파일 형식은 오류로 표시합니다. 문서에 근거가 있어도 추출·검색·모델 판단에 따라 찾지 못할 수 있으므로 중요한 답변은 원문과 함께 확인하세요.

## 대화 저장 및 초기화

대화는 `.chatbot/history.sqlite3`에 저장되어 로컬 새로고침과 서버 재시작 후에도 불러옵니다. 사이드바의 **대화 초기화** 버튼을 누르면 저장된 대화를 삭제합니다. `.chatbot` 폴더는 Git에 업로드하지 않습니다.

현재 대화 기록은 같은 서버를 사용하는 모든 사용자에게 공유됩니다. 공개 서비스에서 개인별 대화를 제공하려면 사용자별 기록 분리가 필요합니다. Streamlit Cloud의 로컬 파일은 재배포나 실행 환경 교체 시 유지가 보장되지 않습니다.

## 데이터 검증

사이드바의 **데이터 검증 실행** 버튼으로 PDF 추출, 문서 내 대표 근거, 파일 변경 여부를 확인할 수 있습니다. **실제 검색/답변 검사**를 선택하면 API를 호출하여 검색 결과와 답변도 검사합니다. 결과는 화면에서 확인하고 JSON으로 다운로드할 수 있습니다.

CLI에서도 실행할 수 있습니다.

```powershell
# 문서 검증: OpenAI API 호출 없음
uv run --locked python -W error scripts/validate_data.py

# 검색과 답변 검증: OpenAI API 사용량 발생
uv run --locked python -W error scripts/validate_data.py --live
```

- 검증 질문과 기대 근거: `validation_cases.json`
- 최신 결과: `.chatbot/validation/latest.json`
- API 또는 실행 오류 결과: `.chatbot/validation/api_failure.json`
- CLI 종료 코드: 정상 완료 `0`, 검증 실패 `1`, 실행·API 오류 `2`

검증 사례의 `expected_terms`는 필수 표현, `any_terms`는 하나 이상 필요한 표현, `forbidden_terms`는 금지 표현, `expected_sources`는 기대 출처를 뜻합니다. `reviewed`는 사람이 기대 근거를 확인한 후 `true`로 설정합니다. 자동 검사 통과만으로 PDF의 모든 표·금액·예외 조항이나 답변의 의미가 검증되는 것은 아닙니다.

## 개발용 점검

다음 명령은 실제 OpenAI 요청 없이 환경과 주요 동작을 확인합니다.

```powershell
uv run --locked python -W error scripts/check_environment.py
uv pip check
uv run --locked python -W error scripts/test_api_key_config.py
uv run --locked python -W error scripts/test_rag_app.py
uv run --locked python -W error scripts/test_chat_history.py
uv run --locked python -W error scripts/test_streamlit_rerun.py
uv run --locked python -W error scripts/test_validation.py
```

실제 API를 이용한 RAG 점검은 별도로 실행합니다.

```powershell
uv run --locked python -W error scripts/test_rag_app.py --live
```

## Streamlit Community Cloud 배포

1. [Streamlit Community Cloud](https://share.streamlit.io)에 로그인하고 **Create app**을 선택합니다.
2. Repository는 `itbank6075-itb/chatbot`, Branch는 `main`, Main file path는 `app.py`로 설정합니다.
3. **Advanced settings**에서 Python 3.11을 선택합니다.
4. **Secrets**에 아래 TOML을 입력하고 값만 실제 키로 바꿉니다.

```toml
OPENAI_API_KEY = "여기에 실제 API 키 입력"
```

5. **Save** 후 **Deploy**를 진행합니다. 배포 후에는 앱의 설정에서 Secrets를 수정할 수 있습니다.

API 키는 GitHub에 업로드하지 않고 Cloud의 Secrets에만 저장합니다. 이 프로젝트는 `uv.lock`과 `pyproject.toml`을 사용합니다. Community Cloud는 `uv.lock`을 지원하므로 별도의 `requirements.txt`는 필요하지 않습니다.

배포 설정과 의존성 지원은 [공식 배포 안내](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [Secrets 관리](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management), [의존성 안내](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)를 참고하세요.

## 문제 해결

| 현상 | 확인할 내용 |
| --- | --- |
| API 키 안내가 표시됨 | 로컬 `.env` 또는 Cloud Secrets의 `OPENAI_API_KEY` 설정 |
| OpenAI 인증 실패 | API 키의 유효성 및 해당 프로젝트 권한 |
| 요청 한도 또는 잔액 오류 | OpenAI API의 사용 한도와 결제·잔액 |
| 근거를 찾지 못함 | 원문 포함 여부, PDF 추출 결과, 대표 질문 검증 결과 |
| PDF에서 텍스트를 읽지 못함 | 스캔 문서 여부 및 별도 OCR 처리 필요 여부 |
| Cloud에서 대화가 사라짐 | 서버 로컬 저장 방식의 한계; 별도 영구 저장소 필요 |

Windows의 한글 경로에서 editable 설치 관련 `UnicodeDecodeError`가 발생하면 다음 스크립트로 `.pth` 경로를 보정할 수 있습니다. 프로젝트를 재설치하거나 폴더를 이동하면 보정이 다시 필요할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -S scripts\repair_venv_path.py
```

uv 동기화 자체가 같은 오류로 실패한다면 기존 가상환경을 먼저 보정하고, `uv sync --locked` 후 다시 보정합니다.

## 프로젝트 구조

```text
chatbot/
├── app.py                         # Streamlit 화면과 RAG 로직
├── DATA/                          # 검색 대상 문서
├── rag_validation.py              # 문서·검색·답변 검증
├── validation_cases.json          # 검증 질문과 기대 근거
├── scripts/                       # 환경 점검 및 테스트
├── pyproject.toml                 # Python 버전과 의존성
├── uv.lock                        # 설치 버전 잠금 파일
├── .env.example                   # API 키 설정 양식 (빈 값)
├── .gitignore                     # 비밀 파일과 로컬 생성물 제외
└── README.md                      # 사용 및 배포 안내
```
