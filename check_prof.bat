@echo off
REM Windows CMD / PowerShell launcher for check_prof
REM Usage: check_prof.bat fm_correlator
REM        check_prof.bat --all
set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%scripts\check_profiling.py" %*
