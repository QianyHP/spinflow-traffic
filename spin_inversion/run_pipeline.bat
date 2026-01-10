@echo off
chcp 65001 >nul
REM ========================================================================
REM Spinflow - One-Click Experiment Runner
REM ========================================================================

echo.
echo ================================================================================
echo Spinflow: Spin-CTM Spin Field Inversion
echo Statistical Physics + Kerner Three-Phase Theory
echo ================================================================================
echo.

REM Set Dataset (Change this to RML or XAM-N6 for other datasets)
set DATASET=YTDJ

REM Clean up old results for this dataset
echo [1/5] Cleaning old results for %DATASET%...
del /Q ..\results\%DATASET%\*.png 2>nul
del /Q ..\results\%DATASET%\*.txt 2>nul
del /Q ..\results\%DATASET%\*.npz 2>nul
echo     Done!
echo.

REM Step 1: Main Inversion
echo [2/5] Running Main Inversion (main.py) for %DATASET%...
echo ================================================================================
python main.py --dataset %DATASET%

if errorlevel 1 (
    echo.
    echo [ERROR] Main Inversion Failed!
    pause
    exit /b 1
)
echo.
echo     Done!
echo.

REM Set NPZ Path
set NPZ_PATH=..\results\%DATASET%\%DATASET%_inverse_init.npz

REM Step 2: Generate Plots
echo [3/5] Generating Plots (viz_diagnostics.py)...
echo ================================================================================
python viz_diagnostics.py %NPZ_PATH%

if errorlevel 1 (
    echo.
    echo [ERROR] Plot Generation Failed!
    pause
    exit /b 1
)
echo.
echo     Done!
echo.

REM Step 3: Evaluation
echo [4/5] Running Evaluation (evaluator.py)...
echo ================================================================================
python evaluator.py %NPZ_PATH%

if errorlevel 1 (
    echo.
    echo [ERROR] Evaluation Failed!
    pause
    exit /b 1
)
echo.
echo     Done!
echo.

REM Step 4: Spacetime Visualization
echo [5/5] Visualizing Spacetime Regions (viz_spacetime.py)...
echo ================================================================================
python viz_spacetime.py %NPZ_PATH%

if errorlevel 1 (
    echo.
    echo [ERROR] Spacetime Visualization Failed!
    pause
    exit /b 1
)
echo.
echo     Done!
echo.

REM Final Summary
echo.
echo ================================================================================
echo [SUCCESS] All Experiments Completed!
echo ================================================================================
echo.
echo Displaying Latest Experiment Reports...
echo.
echo [1/2] %DATASET%_experiment_report.txt (Statistical Summary)
echo --------------------------------------------------------------------------------
type ..\results\%DATASET%\%DATASET%_experiment_report.txt
echo.
echo.
echo [2/2] %DATASET%_report.txt (Quality Report)
echo --------------------------------------------------------------------------------
type ..\results\%DATASET%\%DATASET%_report.txt
echo.
echo ================================================================================
echo Project Ready for Paper Writing!
echo ================================================================================
echo.
pause
