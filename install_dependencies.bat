@echo off
echo ===================================================
echo Intelligent Route Guidance System - Setup
echo ===================================================
echo.
echo Installing required Python packages...
echo.

python -m pip install --upgrade pip --user
pip install -r requirements.txt --user

echo.
echo ===================================================
echo Setup Complete!
echo You can now run the application by double-clicking
echo 'start_server.bat' or running 'python web_app.py'
echo ===================================================
pause
