# ======================================================================
# ファイル名: auto_backup_7z.ps1 (最終修正版)
# 日付: 2025年08月22日
# 役割: プロジェクトを7-Zipで圧縮し、rcloneでGoogle Driveにアップロードする
# ======================================================================

# --- ↓↓↓ 設定項目 (あなたの環境に合わせて確認・修正してください) ↓↓↓ ---

# 1. 7-Zip実行ファイルの場所
$P_7z = "C:\Program Files\7-Zip\7z.exe"

# 2. バックアップ対象のフォルダ
$Src = "C:\Users\USER\tools\atelier-kyo-manager"

# 3. バックアップファイルの保存先と名前
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
# ★★★ ここをDドライブからCドライブの新しいフォルダに変更しました ★★★
$Arch = "C:\Users\USER\Backups\atelier-kyo-manager_$Timestamp.7z"

# 4. rcloneのリモート名とアップロード先フォルダ
$Remote = "gdrive:Backup/sync"

# --- ↑↑↑ 設定項目ここまで ↑↑↑ ---


# --- 実行ログ ---
Write-Host "=================================================="
Write-Host "Atelier Kyo Manager 自動バックアップを開始します..."
Write-Host "バックアップ対象: $Src"
Write-Host "一時保存先: $Arch"
Write-Host "クラウド送信先: $Remote"
Write-Host "=================================================="


# --- 7-Zipでの圧縮処理 ---
if (Test-Path $P_7z) {
    & $P_7z a -t7z -mx=9 -o"$Arch" "$Src"
    if ($?) {
        Write-Host "[OK] 7-Zipでの圧縮に成功しました。"
    } else {
        Write-Warning "[ERROR] 7-Zipでの圧縮に失敗しました。"
        exit 1
    }
} else {
    Write-Warning "[ERROR] 7-Zipが見つかりません: $P_7z"
    exit 1
}


# --- rcloneでのアップロード処理 (日曜日のみ実行) ---
if ((Get-Date).DayOfWeek -eq 'Sunday') {
    Write-Host "本日は日曜日です。クラウドへのアップロードを実行します..."
    if (rclone copy "$Arch" "$Remote" --stats 10s) {
        Write-Host "[CLOUD] Google Drive へアップロード完了"
    } else {
        Write-Warning "[CLOUD] アップロード失敗"
    }
} else {
    Write-Host "本日は日曜日ではないため、クラウドへのアップロードはスキップします。"
}

Write-Host "バックアップ処理が完了しました。"
