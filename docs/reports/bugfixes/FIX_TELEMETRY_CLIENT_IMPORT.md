# TelemetryClient インポートエラーの修正

## 問題

`browser_use_agent.py` で `TelemetryClient` が定義されていないエラーが発生していました。

```
NameError: name 'TelemetryClient' is not defined
```

## 原因

相対インポートのフォールバック処理で `TelemetryClient` と `TelemetryContext` がインポートされていませんでした。

## 修正内容

1. **相対インポートのフォールバックに追加**
   ```python
   from .browser.telemetry import TelemetryService, TelemetryClient, TelemetryContext
   ```

2. **フォールバック処理にモッククラスを追加**
   ```python
   @dataclass
   class TelemetryContext:
       site: str = ""
       query: str = ""
       run_id: Optional[str] = None
       stage: Optional[str] = None
   
   class TelemetryClient:
       def __init__(self, run_context=None, base_dir: str = ""): pass
       async def save_dom(self, page, name, tctx): pass
       async def save_json(self, name, payload, tctx): pass
       async def save_screenshot(self, page, name, tctx): pass
       async def write_fail_snapshot(self, page, reason, tctx, extra=None): pass
   ```

3. **dataclass のインポートを追加**
   ```python
   from dataclasses import dataclass, field
   ```

## 確認

インポートテストが成功しました：
```bash
python -c "from app.agents.browser_use_agent import BrowserUseAgent; print('✓ インポート成功')"
```

## 次のステップ

実ブラウザテストを再実行してください。

