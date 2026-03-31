@echo off
setlocal

REM Run this in the same folder as:
REM - final_batch_queue_stable_paddle2_numberocr_flaghybrid.py
REM - requirements_stable_paddle2.txt

python -m pip install -r requirements_stable_paddle2.txt
if errorlevel 1 goto :error

python -m pip install --upgrade pyinstaller
if errorlevel 1 goto :error

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist maple-guild-participation-ocr.spec del /q maple-guild-participation-ocr.spec

python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --name maple-guild-participation-ocr ^
  --collect-all paddleocr ^
  --collect-all paddle ^
  --collect-all Cython ^
  --collect-submodules Cython ^
  --collect-data Cython ^
  final_batch_queue_stable_paddle2_numberocr_flaghybrid.py
if errorlevel 1 goto :error

echo.
echo Build complete.
echo Output folder: dist\maple-guild-participation-ocr
pause
exit /b 0

:error
echo.
echo Build failed.
pause
exit /b 1
