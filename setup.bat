@echo off
echo Setting Python version to 3.11...
call pyenv local 3.11 2>nul

if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip >nul

echo Synchronizing dependencies...
:: This will quickly ensure all packages in requirements.txt are installed.
:: If they already are, it finishes in seconds.
python -m pip install -r requirements.txt

echo Performing system health check...
:: pip check scans EVERY installed library to ensure there are no version conflicts.
python -m pip check
if %errorlevel% neq 0 (
    echo.
    echo [!] WARNING: Environment corruption or incompatible libraries detected!
    echo [!] Auto-fixing the environment now. This may take a minute...
    echo.
    :: Force a clean reinstall of all strictly pinned libraries in your requirements
    python -m pip install --force-reinstall -r requirements.txt
    
    :: Verify health check passed after auto-fix
    python -m pip check
    if %errorlevel% neq 0 (
        echo [!] ERROR: Could not auto-resolve all conflicts. Please delete the .venv folder and run setup.bat again.
        pause
        exit /b 1
    ) else (
        echo [+] Environment successfully repaired!
    )
) else (
    echo [+] Environment is healthy.
)

echo.
echo Launching application...
python app.py

pause
