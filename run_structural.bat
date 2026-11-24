@echo off
setlocal
cd /d "%~dp0"
call python_env.bat
"%PYTHON_EXE%" -B -X utf8 -m v9.runner --out runs/reproduce_structural
if errorlevel 1 goto failed
echo Completed.
pause
exit /b 0
:failed
echo The operation stopped. Keep the error text for inspection.
pause
exit /b 1
