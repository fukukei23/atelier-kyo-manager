# MONCLER Drission 診断スクリプト実行用 PowerShell スクリプト（WSL経由）
# WSL内のファイルをWindows PowerShellから実行する場合に使用

param(
    [string]$Query = "down jacket",
    [string]$TargetUrl = "",
    [switch]$Headless = $false,
    [int]$Runs = 1,
    [string]$OutBase = "artifacts/moncler_drission"
)

Write-Host "=" * 80
Write-Host "MONCLER Drission 診断スクリプト実行 (WSL経由)"
Write-Host "=" * 80
Write-Host ""

# WSL内のパス
$wslPath = "\\wsl.localhost\Ubuntu\home\yn441611\atelier-kyo-manager"
$wslLinuxPath = "/home/yn441611/atelier-kyo-manager"

Write-Host "WSLパス: $wslPath"
Write-Host ""

# WSL内のPythonを使用するか、WindowsのPythonを使用するか確認
$useWslPython = $true  # デフォルトはWSL内のPythonを使用

# コマンドを構築
if ($useWslPython) {
    # WSL内のPythonを使用
    Write-Host "WSL内のPythonを使用します"
    Write-Host ""
    
    # WSLコマンドを構築（引数を適切にエスケープ）
    $wslCmdParts = @(
        "cd '$wslLinuxPath'"
        "source venv/bin/activate 2>/dev/null || true"
        "python3 scripts/run_moncler_drission_diagnostics.py"
        "--query", "'$Query'"
    )
    
    if ($TargetUrl) {
        $wslCmdParts += "--target_url", "'$TargetUrl'"
    }
    
    if ($Headless) {
        $wslCmdParts += "--headless"
    }
    
    $wslCmdParts += "--runs", "$Runs"
    $wslCmdParts += "--out_base", "'$OutBase'"
    
    $wslCmd = $wslCmdParts -join " "
    
    Write-Host "実行コマンド:"
    Write-Host "  wsl bash -c `"$wslCmd`""
    Write-Host ""
    Write-Host "=" * 80
    Write-Host ""
    
    # WSLコマンドを実行
    wsl bash -c $wslCmd
    
    $exitCode = $LASTEXITCODE
} else {
    # WindowsのPythonを使用（WSL内のファイルにアクセス）
    Write-Host "WindowsのPythonを使用します（WSL内のファイルにアクセス）"
    Write-Host ""
    
    $pythonScript = Join-Path $wslPath "scripts\run_moncler_drission_diagnostics.py"
    
    if (-not (Test-Path $pythonScript)) {
        Write-Host "エラー: スクリプトが見つかりません: $pythonScript"
        exit 1
    }
    
    Write-Host "Pythonスクリプト: $pythonScript"
    
    # Python実行ファイルを検索
    $pythonExe = "python"
    
    # 仮想環境を確認（WSL内の仮想環境は使えないため、Windows側の仮想環境を確認）
    $venvPath = Join-Path $wslPath ".venv"
    $venvScripts = Join-Path $venvPath "Scripts"
    
    if (Test-Path $venvScripts) {
        $pythonExe = Join-Path $venvScripts "python.exe"
        if (Test-Path $pythonExe) {
            Write-Host "仮想環境を検出: $venvPath"
        } else {
            $pythonExe = "python"
        }
    }
    
    Write-Host "Python実行ファイル: $pythonExe"
    
    # コマンドを構築
    $cmdArgs = @(
        $pythonScript
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
    
    # 実行（カレントディレクトリをWSLパスに設定）
    Push-Location $wslPath
    try {
        & $pythonExe $cmdArgs
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "=" * 80
Write-Host "実行完了 (終了コード: $exitCode)"
Write-Host "=" * 80

# 出力ディレクトリを確認
$artifactsDir = Join-Path $wslPath $OutBase.Replace("/", "\")
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

