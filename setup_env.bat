@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   Setting up Face Recognition GPU Virtual Environment
echo ========================================================

if not exist venv (
    echo [1/5] Creating Python virtual environment in .\venv ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Ensure Python 3.10-3.12 is in your PATH.
        pause
        exit /b 1
    )
) else (
    echo [1/5] Existing .\venv found.
)

echo [2/5] Activating virtual environment...
call .\venv\Scripts\activate.bat

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

echo [4/5] Installing all dependencies from requirements.txt...
pip install -r requirements.txt

echo [5/5] Ensuring ONNX Runtime GPU has priority...
pip uninstall -y onnxruntime >nul 2>&1

echo.
echo ========================================================
echo   Verifying GPU & Key Libraries...
echo ========================================================
python -c "import torch, onnxruntime as ort, cv2, insightface, openpyxl; print(f'  PyTorch:       {torch.__version__} (CUDA: {torch.cuda.is_available()})'); print(f'  GPU Device:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'  ONNX Providers: {ort.get_available_providers()}'); print('  InsightFace:   OK'); print('  OpenPyXL:      OK'); print('\n  All systems operational!')"

echo.
echo ========================================================
echo   Setup Complete!
echo.
echo   To run live recognition:
echo     .\venv\Scripts\python.exe app/face_recog.py --checkpoint checkpoints/epoch_0053_mAP_0.5357.pt
echo.
echo   Or with clean attendance reset:
echo     .\venv\Scripts\python.exe app/face_recog.py --checkpoint checkpoints/epoch_0053_mAP_0.5357.pt --reset-attendance
echo ========================================================
pause
