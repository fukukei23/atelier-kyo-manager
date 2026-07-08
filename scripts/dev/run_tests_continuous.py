#!/usr/bin/env python3
"""
テストを2時間継続実行するスクリプト
定期的にテストを実行し、結果を記録
"""

import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


def run_tests():
    """テストを実行して結果を返す"""
    project_root = Path(__file__).parent

    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--maxfail=5"]

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=600,  # 10分タイムアウト
        )
        return result
    except subprocess.TimeoutExpired:
        print("[エラー] テストがタイムアウトしました（10分）")
        return None
    except Exception as e:
        print(f"[エラー] テスト実行エラー: {e}")
        return None


def main():
    duration_hours = 2
    duration_seconds = duration_hours * 3600
    interval_seconds = 300  # 5分ごとに実行

    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=duration_seconds)
    run_count = 0
    success_count = 0
    failure_count = 0

    # 結果ログファイル
    log_file = Path(__file__).parent / f"test_continuous_log_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"

    print("=" * 80)
    print("テスト継続実行（2時間）")
    print("=" * 80)
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"終了予定: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"実行間隔: {interval_seconds}秒（{interval_seconds // 60}分）")
    print(f"ログファイル: {log_file.name}")
    print("=" * 80)
    print()

    log_entries = []
    log_entries.append("=" * 80)
    log_entries.append("テスト継続実行ログ")
    log_entries.append(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_entries.append(f"終了予定: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_entries.append("=" * 80)
    log_entries.append("")

    try:
        while datetime.now() < end_time:
            run_count += 1
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds()
            remaining = (end_time - current_time).total_seconds()

            print()
            print("=" * 80)
            print(f"[実行 {run_count}] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"経過時間: {elapsed / 60:.1f}分 / 残り時間: {remaining / 60:.1f}分")
            print("=" * 80)

            log_entries.append("")
            log_entries.append("=" * 80)
            log_entries.append(f"[実行 {run_count}] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            log_entries.append(f"経過時間: {elapsed / 60:.1f}分 / 残り時間: {remaining / 60:.1f}分")
            log_entries.append("=" * 80)

            result = run_tests()

            if result is None:
                print("[エラー] テスト実行に失敗しました")
                log_entries.append("[エラー] テスト実行に失敗しました")
                failure_count += 1
            else:
                # 結果を表示
                print(result.stdout)
                log_entries.append("STDOUT:")
                log_entries.append(result.stdout)

                if result.stderr:
                    print("STDERR:", result.stderr, file=sys.stderr)
                    log_entries.append("STDERR:")
                    log_entries.append(result.stderr)

                # 成功/失敗をカウント
                if result.returncode == 0:
                    print("✓ テスト成功")
                    log_entries.append("結果: ✓ テスト成功")
                    success_count += 1
                else:
                    print(f"✗ テスト失敗（Return code: {result.returncode}）")
                    log_entries.append(f"結果: ✗ テスト失敗（Return code: {result.returncode}）")
                    failure_count += 1

            # ログを保存
            log_file.write_text("\n".join(log_entries), encoding="utf-8")

            # 次の実行までの待機時間を計算
            if datetime.now() < end_time:
                wait_time = min(interval_seconds, (end_time - datetime.now()).total_seconds())
                if wait_time > 0:
                    print()
                    print(f"[待機] 次の実行まで {wait_time:.0f}秒待機します...")
                    print("（Ctrl+Cで停止できます）")
                    time.sleep(wait_time)

        # 最終サマリー
        print()
        print("=" * 80)
        print("実行完了")
        print("=" * 80)
        print(f"総実行回数: {run_count}")
        print(f"成功: {success_count}")
        print(f"失敗: {failure_count}")
        print(f"ログファイル: {log_file.name}")
        print("=" * 80)

        log_entries.append("")
        log_entries.append("=" * 80)
        log_entries.append("実行完了")
        log_entries.append("=" * 80)
        log_entries.append(f"総実行回数: {run_count}")
        log_entries.append(f"成功: {success_count}")
        log_entries.append(f"失敗: {failure_count}")
        log_entries.append("=" * 80)

        log_file.write_text("\n".join(log_entries), encoding="utf-8")

    except KeyboardInterrupt:
        print()
        print()
        print("=" * 80)
        print("ユーザーによって中断されました")
        print("=" * 80)
        print(f"総実行回数: {run_count}")
        print(f"成功: {success_count}")
        print(f"失敗: {failure_count}")
        print(f"ログファイル: {log_file.name}")
        print("=" * 80)

        log_entries.append("")
        log_entries.append("=" * 80)
        log_entries.append("ユーザーによって中断されました")
        log_entries.append("=" * 80)
        log_entries.append(f"総実行回数: {run_count}")
        log_entries.append(f"成功: {success_count}")
        log_entries.append(f"失敗: {failure_count}")
        log_entries.append("=" * 80)

        log_file.write_text("\n".join(log_entries), encoding="utf-8")

    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
