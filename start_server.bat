@echo off
setlocal
cd /d "%~dp0"

echo.
echo ============================================
echo   AI Hand Energy Mesh Studio
echo ============================================
echo.
echo Starting local server on http://localhost:5500
echo Keep this window open while using the project.
echo Press Ctrl+C to stop the server.
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:5500"
    py -m http.server 5500
    goto :end
)

where python >nul 2>nul
if %errorlevel%==0 (
    start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:5500"
    python -m http.server 5500
    goto :end
)

echo Python was not found.
echo Install Python or use the VS Code Live Server extension.
pause

:end
endlocal









