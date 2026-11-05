@echo off
setlocal
cd /d "%~dp0"
call python_env.bat
"%PYTHON_EXE%" -B -X utf8 tools/run_baseline.py --profile full --out runs/reproduce_baseline
if errorlevel 1 goto failed
"%PYTHON_EXE%" -B -X utf8 -m v8.runner --profile full --out runs/reproduce_robustness
if errorlevel 1 goto failed
"%PYTHON_EXE%" -B -X utf8 -m v9.runner --out runs/reproduce_structural
if errorlevel 1 goto failed
echo Completed.
pause
exit /b 0
:failed
echo The operation stopped. Keep the error text for inspection.
pause
exit /b 1
