@echo off
REM Abre o radar passivo VOR. Argumentos opcionais: --sdr  --sim  --sim-full
cd /d "%~dp0"
".venv\Scripts\python.exe" -m vorradar %*
