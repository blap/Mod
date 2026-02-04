@echo off
setlocal

echo [Inference-PIO] Starting...

REM Check if python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b 1
)

REM Check for virtual environment (optional but recommended)
if exist ".venv" (
    echo [Inference-PIO] Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Check if requirements are installed (rudimentary check)
python -c "import rich" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [Inference-PIO] Installing dependencies...
    pip install -r requirements.txt
)

echo [Inference-PIO] Launching CLI...
python src/inference_pio/__main__.py %*

if %ERRORLEVEL% neq 0 (
    echo [Inference-PIO] Exited with error.
    pause
)

endlocal
