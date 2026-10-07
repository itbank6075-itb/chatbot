"""실제 키나 API 호출 없이 Cloud/로컬 키 읽기와 비밀 파일 제외를 검사합니다."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from streamlit.errors import StreamlitSecretNotFoundError

class MissingSecrets:
    def get(self, key):
        raise StreamlitSecretNotFoundError("No secrets found", error_id="no-secrets-found")

class InvalidSecrets:
    def get(self, key):
        raise StreamlitSecretNotFoundError("SENSITIVE_CONFIG_DO_NOT_PRINT", error_id="failed-parsing-secrets-file")

with TemporaryDirectory() as folder, patch.object(app,"ROOT",Path(folder)):
    (Path(folder)/".env").write_text("OPENAI_API_KEY=local-test-placeholder\n",encoding="utf-8")
    with patch.object(app.st,"secrets",{"OPENAI_API_KEY":"  cloud-test-placeholder  "}):
        assert app.read_api_key()=="cloud-test-placeholder"
    with patch.object(app.st,"secrets",MissingSecrets()):
        assert app.read_api_key()=="local-test-placeholder"
    with patch.object(app.st,"secrets",{}):
        assert app.read_api_key()=="local-test-placeholder"
    with patch.object(app.st,"secrets",{"OPENAI_API_KEY":""}):
        assert app.read_api_key()=="", "Do not silently replace an explicitly empty Cloud key"
    with patch.object(app.st,"secrets",{"OPENAI_API_KEY":123}):
        try:
            app.read_api_key()
        except ValueError:
            pass
        else:
            raise AssertionError("Non-string key must be rejected")
    with patch.object(app.st,"secrets",InvalidSecrets()):
        try:
            app.read_api_key()
        except ValueError as exc:
            assert "SENSITIVE_CONFIG" not in str(exc)
            assert exc.__suppress_context__
        else:
            raise AssertionError("Invalid TOML must not be silently ignored")
    (Path(folder)/".env").unlink()
    with patch.object(app.st,"secrets",MissingSecrets()):
        assert app.read_api_key()==""

root=Path(__file__).resolve().parents[1]
git=r"C:\Program Files\Git\cmd\git.exe"
for filename in (".env",".env.bak",".venv/probe",".streamlit/secrets.toml",".streamlit/secrets.toml.bak"):
    result=subprocess.run([git,"check-ignore","-q","--no-index","--",filename],cwd=root)
    assert result.returncode==0, filename+" must be ignored"
tracked=subprocess.check_output([git,"ls-files","-z"],cwd=root).decode("utf-8").split("\x00")
assert ".env" not in tracked and ".streamlit/secrets.toml" not in tracked
print("PASS: Cloud key, local fallback, missing/empty keys, malformed settings, secret files ignored")
print("No real API key was read or printed; no API request was made.")
