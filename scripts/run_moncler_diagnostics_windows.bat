@echo off
REM MONCLER Drission 診断スクリプト実行用バッチファイル
REM Windows 環境で実行する場合に使用

setlocal enabledelayedexpansion

set QUERY=down jacket
set TARGET_URL=
set HEADLESS=
set RUNS=1
set OUT_BASE=artifacts\moncler_drission

REM 引数の解析
:parse_args
if "%~1"=="" goto :end_parse
if /i "%~1"=="--query" (
    set QUERY=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--target_url" (
    set TARGET_URL=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--headless" (
    set HEADLESS=--headless
    shift
    goto :parse_args
)
if /i "%~1"=="--runs" (
    set RUNS=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--out_base" (
    set OUT_BASE=%~2
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args

:end_parse

echo ================================================================================
echo MONCLER Drission 診断スクリプト実行 (Windows)
echo ================================================================================
echo.

REM プロジェクトルートを取得
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%\.."

echo プロジェクトルート: %CD%
echo.

REM 仮想環境の確認
if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=.venv\Scripts\python.exe
    echo 仮想環境を検出: .venv
) else (
    set PYTHON_EXE=python
    echo 警告: 仮想環境が見つかりません。システムのPythonを使用します。
)

echo.
echo 実行コマンド:
echo   %PYTHON_EXE% scripts\run_moncler_drission_diagnostics.py --query "%QUERY%" %HEADLESS% --runs %RUNS% --out_base "%OUT_BASE%"
if not "%TARGET_URL%"=="" (
    echo   --target_url "%TARGET_URL%"
)
echo.
echo ================================================================================
echo.

REM 実行
%PYTHON_EXE% scripts\run_moncler_drission_diagnostics.py --query "%QUERY%" %HEADLESS% --runs %RUNS% --out_base "%OUT_BASE%"
if not "%TARGET_URL%"=="" (
    %PYTHON_EXE% scripts\run_moncler_drission_diagnostics.py --query "%QUERY%" --target_url "%TARGET_URL%" %HEADLESS% --runs %RUNS% --out_base "%OUT_BASE%"
)

set EXIT_CODE=%ERRORLEVEL%

echo.
echo ================================================================================
echo 実行完了 (終了コード: %EXIT_CODE%)
echo ================================================================================

REM 出力ディレクトリを確認
if exist "%OUT_BASE%" (
    echo.
    echo 診断ディレクトリ: %OUT_BASE%
    dir /b /o-d "%OUT_BASE%" | findstr /r "^[0-9]" >nul
    if %ERRORLEVEL%==0 (
        for /f "delims=" %%d in ('dir /b /o-d "%OUT_BASE%" ^| findstr /r "^[0-9]"') do (
            echo 最新の診断ディレクトリ: %%d
            echo.
            echo 保存されたファイル:
            dir /b "%OUT_BASE%\%%d"
            goto :end_check
        )
    )
)

:end_check
exit /b %EXIT_CODE%

