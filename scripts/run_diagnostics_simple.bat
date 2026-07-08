@echo off
REM 簡単な実行用バッチファイル
REM Windows 環境で実行する場合に使用

echo ================================================================================
echo MONCLER Drission 診断スクリプト実行
echo ================================================================================
echo.

REM プロジェクトルートに移動
cd /d "%~dp0\.."

REM 環境確認スクリプトを実行
echo 環境確認中...
python scripts\check_windows_environment.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo 環境確認に失敗しました。上記のエラーを修正してください。
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo 診断スクリプトを実行します
echo ================================================================================
echo.

REM 仮想環境を確認
if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=.venv\Scripts\python.exe
    echo 仮想環境を使用: .venv
) else (
    set PYTHON_EXE=python
    echo システムのPythonを使用
)

echo.
echo 実行中...
echo.

REM 診断スクリプトを実行
%PYTHON_EXE% scripts\run_moncler_drission_diagnostics.py --query "down jacket" --headless

set EXIT_CODE=%ERRORLEVEL%

echo.
echo ================================================================================
if %EXIT_CODE% equ 0 (
    echo 実行が正常に完了しました
) else (
    echo 実行中にエラーが発生しました (終了コード: %EXIT_CODE%)
)
echo ================================================================================

REM 出力ディレクトリを確認
if exist "artifacts\moncler_drission" (
    echo.
    echo 診断結果は以下のディレクトリに保存されました:
    for /f "delims=" %%d in ('dir /b /o-d "artifacts\moncler_drission" ^| findstr /r "^[0-9]"') do (
        echo   artifacts\moncler_drission\%%d
        goto :end_check
    )
)

:end_check
echo.
pause
exit /b %EXIT_CODE%

