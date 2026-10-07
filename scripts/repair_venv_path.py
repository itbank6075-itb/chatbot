"""Repair uv_build's editable path on Windows with Python 3.11.

Run with python -S so a broken .pth file cannot prevent startup.
"""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
venv = root / ".venv"
if Path(sys.executable).resolve().parent != (venv / "Scripts").resolve():
    raise SystemExit("Run with .venv/Scripts/python.exe -S scripts/repair_venv_path.py")
pth = venv / "Lib" / "site-packages" / "chatbot.pth"
if not pth.is_file():
    raise SystemExit("chatbot.pth not found; synchronize the project first.")
pth.write_text(
    "import sys; sys.path.insert(0, " + ascii(str(root / "src")) + ")\n",
    encoding="ascii",
)
print("Editable project path repaired.")
