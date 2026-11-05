@echo off
setlocal
cd /d "%~dp0"
call python_env.bat
"%PYTHON_EXE%" -B -X utf8 -m v8.environment
if errorlevel 1 goto failed
echo Completed.
pause
exit /b 0
:failed
echo The operation stopped with an error. Keep this window open and copy the error text.
pause
exit /b 1
