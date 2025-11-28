# Cursorのチャット履歴データベースファイルを検索するPowerShellスクリプト
# 使い方: PowerShellで実行
#   .\tools\find_cursor_chat_history.ps1

$cursorProjectsDir = "$env:USERPROFILE\.cursor\projects"

Write-Host "=" * 80
Write-Host "Cursor チャット履歴検索ツール"
Write-Host "=" * 80
Write-Host ""

if (-not (Test-Path $cursorProjectsDir)) {
    Write-Host "❌ Cursorプロジェクトディレクトリが見つかりません: $cursorProjectsDir" -ForegroundColor Red
    Write-Host ""
    Write-Host "手動で確認してください:"
    Write-Host "  1. Windowsのエクスプローラーを開く"
    Write-Host "  2. アドレスバーに以下を入力:"
    Write-Host "     $cursorProjectsDir"
    exit
}

Write-Host "📁 プロジェクトディレクトリ: $cursorProjectsDir"
Write-Host ""

# すべてのプロジェクトをリストアップ
$projects = Get-ChildItem -Path $cursorProjectsDir -Directory -ErrorAction SilentlyContinue

if (-not $projects) {
    Write-Host "⚠️  プロジェクトディレクトリが見つかりませんでした" -ForegroundColor Yellow
    exit
}

Write-Host "📦 $($projects.Count) 個のプロジェクトが見つかりました:"
$projects | ForEach-Object { Write-Host "   - $($_.Name)" }
Write-Host ""
Write-Host "=" * 80
Write-Host ""

# 各プロジェクトを分析
$allDbFiles = @()
$cutoffTime = (Get-Date).AddDays(-1).Date  # 昨日の0時以降

Write-Host "⏰ 検索条件: 更新日時が $($cutoffTime.ToString('yyyy-MM-dd HH:mm:ss')) 以降のファイルを優先表示" -ForegroundColor Cyan
Write-Host ""

foreach ($project in $projects) {
    Write-Host "🔍 分析中: $($project.Name)"
    
    # データベースファイルを検索
    $dbFiles = Get-ChildItem -Path $project.FullName -Recurse -Filter "*.db" -ErrorAction SilentlyContinue | 
               Where-Object { -not $_.PSIsContainer }
    
    # チャット履歴に関連しそうなファイルも検索
    $chatDbFiles = Get-ChildItem -Path $project.FullName -Recurse -Filter "*chat*.db" -ErrorAction SilentlyContinue | 
                   Where-Object { -not $_.PSIsContainer }
    
    $historyDbFiles = Get-ChildItem -Path $project.FullName -Recurse -Filter "*history*.db" -ErrorAction SilentlyContinue | 
                      Where-Object { -not $_.PSIsContainer }
    
    # workspaceStorage ディレクトリを検索
    $storageDirs = Get-ChildItem -Path $project.FullName -Recurse -Directory -Filter "workspaceStorage" -ErrorAction SilentlyContinue
    
    $allFiles = $dbFiles + $chatDbFiles + $historyDbFiles | Sort-Object -Unique -Property FullName
    
    if ($allFiles) {
        $recentFiles = $allFiles | Where-Object { $_.LastWriteTime -ge $cutoffTime }
        Write-Host "   ✅ $($allFiles.Count) 個のデータベースファイルが見つかりました" -ForegroundColor Green
        if ($recentFiles) {
            Write-Host "   🆕 そのうち $($recentFiles.Count) 個が最近（昨日以降）更新されています" -ForegroundColor Yellow
        }
        $allDbFiles += $allFiles
    } else {
        Write-Host "   ⚠️  データベースファイルが見つかりませんでした" -ForegroundColor Yellow
    }
    
    if ($storageDirs) {
        Write-Host "   📂 workspaceStorage: $($storageDirs.Count) 個" -ForegroundColor Cyan
    }
    Write-Host ""
}

# 結果をまとめて表示
Write-Host "=" * 80
Write-Host "検索結果（更新日時順・最近更新されたものが先頭）"
Write-Host "=" * 80
Write-Host ""

$resultsByProject = $allDbFiles | Group-Object { 
    $pathParts = $_.FullName.Replace($cursorProjectsDir + "\", "").Split('\')
    if ($pathParts.Length -gt 0) { $pathParts[0] } else { "Unknown" }
}

foreach ($group in $resultsByProject) {
    Write-Host "📦 プロジェクト: $($group.Name)" -ForegroundColor Cyan
    Write-Host "   パス: $cursorProjectsDir\$($group.Name)"
    Write-Host ""
    
    # 更新日時でソート（最近更新されたものから）
    $sortedFiles = $group.Group | Sort-Object -Property LastWriteTime -Descending
    
    # 最近更新されたファイルを強調表示
    $recentFiles = $sortedFiles | Where-Object { $_.LastWriteTime -ge $cutoffTime }
    $oldFiles = $sortedFiles | Where-Object { $_.LastWriteTime -lt $cutoffTime }
    
    if ($recentFiles) {
        Write-Host "   🆕 最近更新されたファイル ($($recentFiles.Count) 個):" -ForegroundColor Yellow
        foreach ($file in $recentFiles) {
            $sizeKB = [math]::Round($file.Length / 1KB, 2)
            $sizeMB = [math]::Round($file.Length / 1MB, 2)
            $sizeStr = if ($file.Length -gt 1MB) { "$sizeMB MB" } else { "$sizeKB KB" }
            
            $timeStr = $file.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
            $isRecent = $file.LastWriteTime -ge (Get-Date).AddHours(-12)
            
            Write-Host "      ⭐ $($file.Name)" -ForegroundColor $(if ($isRecent) { "Green" } else { "Yellow" })
            Write-Host "         サイズ: $sizeStr"
            Write-Host "         更新日時: $timeStr $(if ($isRecent) { '(今朝以降)' } else { '(昨日以降)' })" -ForegroundColor $(if ($isRecent) { "Green" } else { "Yellow" })
            Write-Host "         パス: $($file.FullName)"
            
            # 相対パスも表示
            $relativePath = $file.FullName.Replace($cursorProjectsDir + "\", "")
            Write-Host "         相対パス: $relativePath"
            Write-Host ""
        }
    }
    
    if ($oldFiles) {
        Write-Host "   📄 それ以前のファイル ($($oldFiles.Count) 個):" -ForegroundColor Gray
        foreach ($file in $oldFiles | Select-Object -First 5) {
            $sizeKB = [math]::Round($file.Length / 1KB, 2)
            $sizeMB = [math]::Round($file.Length / 1MB, 2)
            $sizeStr = if ($file.Length -gt 1MB) { "$sizeMB MB" } else { "$sizeKB KB" }
            
            Write-Host "      • $($file.Name)" -ForegroundColor DarkGray
            Write-Host "         サイズ: $sizeStr"
            Write-Host "         更新日時: $($file.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))"
            Write-Host ""
        }
        if ($oldFiles.Count -gt 5) {
            Write-Host "      ... 他 $($oldFiles.Count - 5) 個のファイル" -ForegroundColor DarkGray
        }
    }
    
    Write-Host "-" * 80
    Write-Host ""
}

# 最近更新されたファイルを優先表示
$recentFiles = $allDbFiles | Where-Object { $_.LastWriteTime -ge $cutoffTime }

if ($recentFiles) {
    Write-Host "🎯 最近更新されたデータベースファイル（再起動後のチャット履歴の可能性が高い）:" -ForegroundColor Green
    Write-Host ""
    
    # 更新日時順にソート
    $sortedRecent = $recentFiles | Sort-Object -Property LastWriteTime -Descending | Select-Object -First 10
    
    foreach ($file in $sortedRecent) {
        $sizeKB = [math]::Round($file.Length / 1KB, 2)
        $sizeMB = [math]::Round($file.Length / 1MB, 2)
        $sizeStr = if ($file.Length -gt 1MB) { "$sizeMB MB" } else { "$sizeKB KB" }
        
        $timeStr = $file.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
        $projectName = $file.FullName.Replace($cursorProjectsDir + "\", "").Split('\')[0]
        
        Write-Host "   ⭐ プロジェクト: $projectName" -ForegroundColor Cyan
        Write-Host "      ファイル: $($file.Name)"
        Write-Host "      サイズ: $sizeStr"
        Write-Host "      更新日時: $timeStr"
        Write-Host "      パス: $($file.FullName)"
        Write-Host ""
    }
}

# 最大サイズのファイル（参考）
if ($allDbFiles) {
    $largest = $allDbFiles | Sort-Object -Property Length -Descending | Select-Object -First 1
    $sizeKB = [math]::Round($largest.Length / 1KB, 2)
    $sizeMB = [math]::Round($largest.Length / 1MB, 2)
    $sizeStr = if ($largest.Length -gt 1MB) { "$sizeMB MB" } else { "$sizeKB KB" }
    
    Write-Host "💡 最も大きなデータベースファイル（参考）:" -ForegroundColor Gray
    Write-Host "   ファイル: $($largest.Name)"
    Write-Host "   サイズ: $sizeStr"
    Write-Host "   更新日時: $($largest.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))"
    Write-Host "   パス: $($largest.FullName)"
    Write-Host ""
}

Write-Host "=" * 80
Write-Host "完了"
Write-Host "=" * 80
Write-Host ""
Write-Host "💡 ヒント:"
Write-Host "   - 大きいデータベースファイルがチャット履歴の可能性が高いです"
Write-Host "   - workspaceStorage ディレクトリ内にも履歴がある可能性があります"
Write-Host "   - プロジェクトを開く際は、常に同じパスから開くようにしてください"
Write-Host ""

