@echo off
setlocal
cd /d "%~dp0"
call python_env.bat
"%PYTHON_EXE%" -B -X utf8 -m v8.environment --python-only
if errorlevel 1 goto failed
if not exist ".venv\Scripts\python.exe" "%PYTHON_EXE%" -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -B -X utf8 -m v8.environment
if errorlevel 1 goto failed
echo Completed.
pause
exit /b 0
:failed
echo The operation stopped with an error. Keep this window open and copy the error text.
pause
exit /b 1
