# PowerShell automated environment setup
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Setting up Face Recognition GPU Virtual Environment" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

if (-not (Test-Path "venv")) {
    Write-Host "[1/5] Creating Python virtual environment in .\venv ..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create venv. Ensure Python 3.10-3.12 is in your PATH."
        exit 1
    }
} else {
    Write-Host "[1/5] Existing .\venv found." -ForegroundColor Green
}

Write-Host "[2/5] Upgrading pip..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "[3/5] Installing all dependencies from requirements.txt..." -ForegroundColor Yellow
.\venv\Scripts\pip.exe install -r requirements.txt

Write-Host "[4/5] Ensuring ONNX Runtime GPU has priority..." -ForegroundColor Yellow
.\venv\Scripts\pip.exe uninstall -y onnxruntime 2>$null

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "  Verifying GPU & Key Libraries..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
.\venv\Scripts\python.exe -c @"
import torch, onnxruntime as ort, cv2, insightface, openpyxl
print(f'  PyTorch:       {torch.__version__} (CUDA: {torch.cuda.is_available()})')
if torch.cuda.is_available():
    print(f'  GPU Device:    {torch.cuda.get_device_name(0)}')
print(f'  ONNX Providers: {ort.get_available_providers()}')
print('  InsightFace:   OK')
print('  OpenPyXL:      OK')
print('\n  All systems operational!')
"@

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "  To run live recognition:" -ForegroundColor White
Write-Host "    .\venv\Scripts\python.exe app/face_recog.py --checkpoint checkpoints/epoch_0053_mAP_0.5357.pt" -ForegroundColor Cyan
Write-Host "  Or with fresh attendance:" -ForegroundColor White
Write-Host "    .\venv\Scripts\python.exe app/face_recog.py --checkpoint checkpoints/epoch_0053_mAP_0.5357.pt --reset-attendance" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Green
