@echo off
REM ========================================================================
REM YTDJ Inversion - Standard Experiment Runner
REM Uses optimized parameters with comprehensive analysis
REM ========================================================================

echo.
echo ========================================================================
echo YTDJ Trajectory Inversion - Standard Experiment
echo ========================================================================
echo.
echo Configuration:
echo   - Road length: 280m (actual data coverage)
echo   - Spatial resolution: 2.00m/cell (140 cells)
echo   - Time resolution: 0.25s
echo   - Bottleneck coverage: 19 cells (80-118m)
echo.
echo ========================================================================
echo.

REM Ensure results directory exists
if not exist results mkdir results

echo [Step 1/2] Running inversion...
echo ========================================================================
python ytdj_inverse_init.py --csv data/VTDJ_6-10.csv --direction eb --road-length 280 --cells 140 --dt-obs 0.25 --t0 60 --T-obs 60 --use-sle --lanes all --group-size 5 --K-inv 80 --iters 60 --use-spsa --save-prefix results/optimal

if errorlevel 1 (
    echo.
    echo ERROR: Inversion failed!
    pause
    exit /b 1
)

echo.
echo [Step 2/2] Generating analysis plots...
echo ========================================================================
python generate_analysis_plots.py

if errorlevel 1 (
    echo.
    echo ERROR: Plot generation failed!
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo SUCCESS: All experiments completed!
echo ========================================================================
echo.
echo Generated files in results/:
echo   Core Data:
echo     - optimal_inverse_init.npz
echo.
echo   Analysis Plots:
echo     - density_comparison.png
echo     - initial_analysis.png
echo     - bottleneck_analysis.png
echo     - experiment_report.txt
echo.
echo ========================================================================
pause


