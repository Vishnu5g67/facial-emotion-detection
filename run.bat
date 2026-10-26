@echo off
title Facial Emotion Detection
echo ========================================================
echo         Facial Emotion Detection System
echo ========================================================
echo.
echo Launching Real-time Webcam Detector...
echo (Press 'q' in the camera window to exit)
echo.

python "%~dp0app.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred while running the application.
    pause
)
