@echo off
REM Wrapper script to run train_sweep.py with the correct Python environment

cd /d "%~dp0.."
call venv\Scripts\activate.bat
set PYTHONPATH=src
python src\mlops_project\train_sweep.py %*
