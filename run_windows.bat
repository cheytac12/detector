@echo off
setlocal enabledelayedexpansion
title Sign Language Detection App

echo +------------------------------------------+
echo ^|   Sign Language Detection App            ^|
echo +------------------------------------------+
echo.

:: ── 0. Auto-apply pyenv-win local version if .python-version exists ──────────
if exist ".python-version" (
    set /p PYENV_VER=<.python-version
    :: Strip carriage return / spaces
    for /f "tokens=* delims= " %%a in ("!PYENV_VER!") do set PYENV_VER=%%a

    echo   Found .python-version: !PYENV_VER!

    where pyenv >nul 2>&1
    if not errorlevel 1 (
        echo   Activating pyenv Python !PYENV_VER! ...
        pyenv local !PYENV_VER! >nul 2>&1
        :: Use 'pyenv root' but capture failure gracefully (pyenv-win may not support it)
        for /f "tokens=*" %%p in ('pyenv root 2^>nul') do set PYENV_ROOT=%%p
        if defined PYENV_ROOT (
            set PATH=!PYENV_ROOT!\shims;!PYENV_ROOT!\versions\!PYENV_VER!\bin;!PATH!
            echo   pyenv shims prepended to PATH.
        ) else (
            :: pyenv root not supported (some pyenv-win builds); use USERPROFILE fallback
            if exist "!USERPROFILE!\.pyenv\pyenv-win\shims" (
                set PYENV_ROOT=!USERPROFILE!\.pyenv\pyenv-win
                set PATH=!PYENV_ROOT!\shims;!PYENV_ROOT!\versions\!PYENV_VER!\bin;!PATH!
                echo   pyenv-win shims prepended to PATH (fallback path).
            ) else (
                echo   pyenv found but root path unknown - continuing without shim injection.
            )
        )
    ) else (
        echo   pyenv-win not in PATH - skipping pyenv activation.
        echo   If Python !PYENV_VER! is not found below, run: pyenv install !PYENV_VER!
    )
)

:: ── 1. Find a suitable Python ────────────────────────────────────────────────
set PYTHON=
set MINOR=

:: Check plain 'python' first (covers pyenv-win shim, conda, direct install)
for %%cmd in (python python3) do (
    if "!PYTHON!"=="" (
        where %%cmd >nul 2>&1
        if not errorlevel 1 (
            for /f "tokens=*" %%r in ('%%cmd -c "import sys; v=sys.version_info; print(\"ok\" if (v.major==3 and 9<=v.minor<=12) else \"bad\")" 2^>nul') do (
                if "%%r"=="ok" (
                    set PYTHON=%%cmd
                    for /f "tokens=*" %%m in ('%%cmd -c "import sys; print(sys.version_info.minor)" 2^>nul') do set MINOR=%%m
                )
            )
        )
    )
)

:: Fallback: try py.exe launcher (standard python.org installer)
if "!PYTHON!"=="" (
    where py >nul 2>&1
    if not errorlevel 1 (
        for %%v in (3.11 3.10 3.9 3.12) do (
            if "!PYTHON!"=="" (
                py -%%v --version >nul 2>&1
                if not errorlevel 1 (
                    set PYTHON=py -%%v
                    for /f "tokens=2 delims=." %%m in ("%%v") do set MINOR=%%m
                )
            )
        )
    )
)

if "!PYTHON!"=="" (
    echo.
    echo [ERROR] No compatible Python found ^(need 3.9, 3.10, 3.11, or 3.12^).
    echo.
    echo   Your system may have Python 3.7 or 3.8 which is too old for mediapipe.
    echo.
    echo   FIX A - If you use pyenv-win:
    echo     pyenv install 3.11.9
    echo     pyenv local 3.11.9
    echo     Then re-run this script.
    echo.
    echo   FIX B - Install Python 3.11 directly:
    echo     https://www.python.org/downloads/release/python-3119/
    echo     During setup, check "Add Python to PATH" and "for all users".
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('!PYTHON! -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\")" 2^>nul') do set PY_FULL=%%v
echo   Using Python !PY_FULL!  ^(!PYTHON!^)
echo.

:: ── 2. Virtual environment ────────────────────────────────────────────────────
if not exist ".venv" (
    echo   Creating virtual environment...
    !PYTHON! -m venv .venv
    if errorlevel 1 (
        echo [ERROR] venv creation failed.
        echo   If using pyenv-win, run:  pyenv rehash
        echo   Then try again.
        pause & exit /b 1
    )
)

set PIP=.venv\Scripts\pip.exe
set PY=.venv\Scripts\python.exe

:: Detect corrupted venv
if not exist "!PY!" (
    echo   Venv appears corrupted - rebuilding...
    rmdir /s /q .venv
    !PYTHON! -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Still cannot create venv.
        pause & exit /b 1
    )
)

:: ── 3. Upgrade pip ───────────────────────────────────────────────────────────
echo   Upgrading pip...
"!PY!" -m pip install --quiet --upgrade pip 2>nul

:: ── 4. Install dependencies ──────────────────────────────────────────────────
echo   Installing dependencies (first run may take several minutes)...

if !MINOR! GEQ 12 (
    echo   Python 3.12 detected - using tensorflow 2.16+ and keras 3
    "!PIP!" install --quiet "tensorflow>=2.16.0" "mediapipe==0.10.21" "opencv-python==4.9.0.80" "opencv-contrib-python==4.9.0.80" "numpy==1.26.4" "Pillow==10.4.0" "scikit-learn==1.4.2" "scipy==1.13.1" "protobuf>=4.25.3,<5.0.0"
) else (
    "!PIP!" install --quiet -r requirements.txt
    if errorlevel 1 (
        echo   Bulk install failed - falling back to individual installs...
        for %%p in ("tensorflow==2.15.1" "mediapipe==0.10.21" "opencv-python==4.9.0.80" "opencv-contrib-python==4.9.0.80" "numpy==1.26.4" "Pillow==10.4.0" "scikit-learn==1.4.2" "scipy==1.13.1" "protobuf>=4.25.3,<5.0.0") do (
            "!PIP!" install --quiet %%p
        )
    )
)

:: ── 5. Mediapipe sanity check ────────────────────────────────────────────────
echo   Checking mediapipe...
"!PY!" -c "import mediapipe; import cv2; import numpy" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Core import check failed. Trying targeted fix...
    "!PIP!" install --quiet --force-reinstall "mediapipe==0.10.21" "numpy==1.26.4" "protobuf>=4.25.3,<5.0.0"
    "!PY!" -c "import mediapipe; import cv2; import numpy" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] mediapipe/cv2/numpy still failing after reinstall.
        echo   Run this to diagnose:
        echo     .venv\Scripts\python.exe -c "import mediapipe"
        pause & exit /b 1
    )
)
echo   Core imports OK.
echo.

:: ── 6. Copy model/data from parent if available ──────────────────────────────
if exist "..\model.h5" (
    if not exist "model.h5" (
        copy "..\model.h5" "model.h5" >nul
        echo   Copied model.h5 from parent directory.
    )
)
if exist "..\model_weights.h5" (
    if not exist "model_weights.h5" (
        copy "..\model_weights.h5" "model_weights.h5" >nul
        echo   Copied model_weights.h5 from parent directory.
    )
)
if exist "..\MP_Data" (
    if not exist "MP_Data" (
        xcopy /E /I /Q "..\MP_Data" "MP_Data" >nul
        echo   Copied MP_Data/ from parent directory.
    )
)

:: ── 7. Launch ────────────────────────────────────────────────────────────────
echo   Starting Sign Language Detection App...
echo.
"!PY!" app.py

if errorlevel 1 (
    echo.
    echo [ERROR] app.py exited with error code %errorlevel%.
    echo   Check output above. Common causes:
    echo     - No camera found: edit app_settings.json, change "camera_source" to 1 or 2
    echo     - model.h5 missing: place model.h5 in this folder
    echo     - tkinter missing: reinstall Python from python.org with tcl/tk checked
    echo     - mediapipe error: run  .venv\Scripts\python.exe -c "import mediapipe"
    echo.
    pause
)
