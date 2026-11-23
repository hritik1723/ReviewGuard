@echo off
echo ==================================================
echo  ReviewGuard: Fake Product Review Detection
echo  Classical ML Intelligence Core
echo ==================================================
python -m uvicorn reviewguard.app.main:app --host 0.0.0.0 --port 8000
