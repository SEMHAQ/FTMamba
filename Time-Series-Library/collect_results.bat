@echo off
REM ============================================
REM Collect and summarize all experiment results
REM ============================================

echo ==========================================
echo  Collecting Multi-seed FTMamba Results
echo ==========================================
echo.
echo --- Seed 2021 ---
findstr /C:"mse:" result_long_term_forecast.txt | findstr /C:"seed2021"
echo.
echo --- Seed 42 ---
findstr /C:"mse:" result_long_term_forecast.txt | findstr /C:"seed42"
echo.
echo --- Seed 1234 ---
findstr /C:"mse:" result_long_term_forecast.txt | findstr /C:"seed1234"

echo.
echo ==========================================
echo  Collecting Weather Results
echo ==========================================
echo.
findstr /C:"mse:" result_long_term_forecast.txt | findstr /C:"weather"

echo.
echo ==========================================
echo  Done! Copy relevant lines to peer_reviews/results_summary.md
echo ==========================================
pause
