@echo off
setlocal enabledelayedexpansion

set "MACHINE_NAME=%1"
if "%MACHINE_NAME%"=="" set "MACHINE_NAME=Machine_%USERNAME%"

for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set "TIMESTAMP=%mydate%_%mytime%"

set "OUTPUT_DIR=results_OpenMP_%MACHINE_NAME%_%TIMESTAMP%"

mkdir "%OUTPUT_DIR%" 2>nul

echo Hostname: %COMPUTERNAME% > "%OUTPUT_DIR%\system_info.txt"
echo Date: %date% %time% >> "%OUTPUT_DIR%\system_info.txt"
echo CPU: >> "%OUTPUT_DIR%\system_info.txt"
wmic cpu get name /format:list | find "Name=" >> "%OUTPUT_DIR%\system_info.txt"
echo Cores: %NUMBER_OF_PROCESSORS% >> "%OUTPUT_DIR%\system_info.txt"

type "%OUTPUT_DIR%\system_info.txt"

set "BASE_DIR=%~dp0"
cd /d "%BASE_DIR%"

cd openmp-naive
call mingw32-make clean
call mingw32-make

if %ERRORLEVEL% EQU 0 (
    (
        echo === OpenMP Naive ===
        for %%s in (100 1000 10000) do (
            echo Size: %%sx%%s
            for %%t in (1 2 4 8 16) do (
                echo Threads: %%t
                if %%s LEQ 1000 (
                    main.exe %%s 1 %%t 128
                ) else (
                    main.exe %%s 0 %%t 128
                )
            )
        )
        echo.
        echo === Testing Z-Order Blocked Method ===
        for %%t in (4 8) do (
            echo Threads: %%t, Block: 64
            main.exe 1000 0 %%t 64
        )
    ) > "..\%OUTPUT_DIR%\openmp_naive_results.txt" 2>&1
)

cd /d "%BASE_DIR%"

cd openmp-strassen
call mingw32-make clean
call mingw32-make

if %ERRORLEVEL% EQU 0 (
    (
        echo === OpenMP Strassen ===
        for %%s in (100 1000 10000) do (
            echo Size: %%sx%%s
            for %%t in (7 14 21 28) do (
                echo Threads: %%t
                if %%s LEQ 1000 (
                    optimized_main.exe %%s 1 %%t 128
                ) else (
                    optimized_main.exe %%s 0 %%t 128
                )
            )
        )
    ) > "..\%OUTPUT_DIR%\openmp_strassen_results.txt" 2>&1
)

cd /d "%BASE_DIR%"

(
    echo === Summary ===
    findstr /R "Size Threads Total.*execution.*time PASSED FAILED" "%OUTPUT_DIR%\openmp_naive_results.txt" 2>nul
    findstr /R "Size Threads Total.*execution.*time PASSED FAILED" "%OUTPUT_DIR%\openmp_strassen_results.txt" 2>nul
) > "%OUTPUT_DIR%\SUMMARY.txt"

echo.
echo Results: %OUTPUT_DIR%\
dir "%OUTPUT_DIR%"

endlocal
