@echo off
@chcp 65001 >nul
REM Run from repo root: scripts\windows\run_pipeline.bat [DATASET]
REM Or: cd scripts\windows && run_pipeline.bat YTDJ

set DATASET=YTDJ
if not "%~1"=="" set DATASET=%~1

pushd %~dp0..\..
set REPO=%CD%

echo [1/5] Cleaning old results for %DATASET%...
del /Q "%REPO%\results\%DATASET%\*.png" 2>nul
del /Q "%REPO%\results\%DATASET%\*.txt" 2>nul
del /Q "%REPO%\results\%DATASET%\*.npz" 2>nul

echo [2/5] Main inversion...
set PYTHONPATH=%REPO%\src
python -m spinflow --dataset %DATASET%
if errorlevel 1 goto :fail

set NPZ_PATH=%REPO%\results\%DATASET%\%DATASET%_inverse_init.npz

echo [3/5] Diagnostics...
python "%REPO%\scripts\viz_diagnostics.py" "%NPZ_PATH%"
if errorlevel 1 goto :fail

echo [4/5] Evaluation...
python "%REPO%\scripts\evaluator.py" "%NPZ_PATH%"
if errorlevel 1 goto :fail

echo [5/5] Spacetime figure...
python "%REPO%\scripts\viz_spacetime.py" "%NPZ_PATH%"
if errorlevel 1 goto :fail

echo.
echo SUCCESS. Report:
type "%REPO%\results\%DATASET%\%DATASET%_report.txt"
popd
exit /b 0

:fail
echo PIPELINE FAILED
popd
exit /b 1
