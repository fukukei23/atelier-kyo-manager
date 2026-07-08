# MONCLER Drission 診断スクリプト実行用 PowerShell スクリプト
# Windows 環境で実行する場合に使用

param(
    [string]$Query = "down jacket",
    [string]$TargetUrl = "",
    [switch]$Headless = $false,
    [int]$Runs = 1,
    [string]$OutBase = "artifacts\moncler_drission"
)

Write-Host "=" * 80
Write-Host "MONCLER Drission 診断スクリプト実行 (Windows)"
Write-Host "=" * 80
Write-Host ""

# プロジェクトルートを取得
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath

Write-Host "プロジェクトルート: $projectRoot"
Write-Host ""

# 仮想環境の確認
$venvPath = Join-Path $projectRoot ".venv"
$venvScripts = Join-Path $venvPath "Scripts"

if (Test-Path $venvScripts) {
    Write-Host "仮想環境を検出: $venvPath"
    $pythonExe = Join-Path $venvScripts "python.exe"
    
    if (Test-Path $pythonExe) {
        Write-Host "Python実行ファイル: $pythonExe"
    } else {
        Write-Host "警告: Python実行ファイルが見つかりません。システムのPythonを使用します。"
        $pythonExe = "python"
    }
} else {
    Write-Host "警告: 仮想環境が見つかりません。システムのPythonを使用します。"
    $pythonExe = "python"
}

Write-Host ""

# 診断スクリプトのパス
$diagnosticsScript = Join-Path $scriptPath "run_moncler_drission_diagnostics.py"

if (-not (Test-Path $diagnosticsScript)) {
    Write-Host "エラー: 診断スクリプトが見つかりません: $diagnosticsScript"
    exit 1
}

# コマンドを構築
$cmdArgs = @(
    $diagnosticsScript
    "--query", $Query
)

if ($TargetUrl) {
    $cmdArgs += "--target_url", $TargetUrl
}

if ($Headless) {
    $cmdArgs += "--headless"
}

$cmdArgs += "--runs", $Runs.ToString()
$cmdArgs += "--out_base", $OutBase

Write-Host "実行コマンド:"
Write-Host "  $pythonExe $($cmdArgs -join ' ')"
Write-Host ""
Write-Host "=" * 80
Write-Host ""

# 実行
Set-Location $projectRoot
& $pythonExe $cmdArgs

$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "=" * 80
Write-Host "実行完了 (終了コード: $exitCode)"
Write-Host "=" * 80

# 出力ディレクトリを確認
$artifactsDir = Join-Path $projectRoot $OutBase
if (Test-Path $artifactsDir) {
    $diagDirs = Get-ChildItem -Path $artifactsDir -Directory | Sort-Object LastWriteTime -Descending
    if ($diagDirs) {
        $latestDir = $diagDirs[0]
        Write-Host ""
        Write-Host "最新の診断ディレクトリ: $($latestDir.FullName)"
        $files = Get-ChildItem -Path $latestDir.FullName -File
        Write-Host "ファイル数: $($files.Count)"
        Write-Host ""
        Write-Host "保存されたファイル:"
        foreach ($file in $files) {
            Write-Host "  - $($file.Name)"
        }
    }
}

exit $exitCode

