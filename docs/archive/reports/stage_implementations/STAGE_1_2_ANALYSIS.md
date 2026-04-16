# Stage 1.2 完全完了 - 現状分析レポート

## Step 1: 現状の重複ポイントの洗い出し

### 1. BrowserUseAgent 内のメソッド定義と利用箇所

#### `_build_context_options` (1394行目)
- **定義**: `app/agents/browser_use_agent.py:1394-1438`
- **利用箇所**: **見つからず**（未使用の可能性）
- **責務**: Playwright BrowserContext 用のオプション構築（viewport, locale, headers, user_agent, HAR/trace/video設定）

#### `_setup_routes` (1440行目)
- **定義**: `app/agents/browser_use_agent.py:1440-1471`
- **利用箇所**: **見つからず**（未使用の可能性）
- **責務**: ルートハンドラーの設定（外部リソースブロック、ロケール書き換え）

#### `_apply_saved_session` (1482行目)
- **定義**: `app/agents/browser_use_agent.py:1482-1530`
- **利用箇所**: **見つからず**（未使用の可能性）
- **責務**: 保存されたCookie/LocalStorageの復元

#### `_setup_init_scripts` (1569行目)
- **定義**: `app/agents/browser_use_agent.py:1569-1650+`
- **利用箇所**: **見つからず**（未使用の可能性）
- **責務**: ブラウザコンテキストの初期化スクリプト設定（ロケール、タイムゾーン、ステルスパッチ）

#### `_get_session_file` (1473行目)
- **定義**: `app/agents/browser_use_agent.py:1473-1480`
- **利用箇所**: `_apply_saved_session` 内で使用
- **責務**: セッションファイルパスの取得

### 2. SessionManager 側の実装

#### `_build_context_options` (291行目)
- **定義**: `app/agents/browser/session_manager.py:291-324`
- **利用箇所**: `_ensure_open` 内（236行目）
- **責務**: BrowserUseAgent版と同等の機能

#### `_setup_routes` (326行目)
- **定義**: `app/agents/browser/session_manager.py:326-386`
- **利用箇所**: `_ensure_open` 内（259行目）
- **責務**: BrowserUseAgent版と同等の機能（URL正規化は `_normalize_url` メソッド経由）

#### `_setup_init_scripts` (396行目)
- **定義**: `app/agents/browser/session_manager.py:396-450+`
- **利用箇所**: `_ensure_open` 内（260行目）
- **責務**: BrowserUseAgent版と同等の機能

#### `_apply_saved_session` (587行目)
- **定義**: `app/agents/browser/session_manager.py:587-624`
- **利用箇所**: `_ensure_open` 内（279行目）
- **責務**: BrowserUseAgent版と同等の機能

#### `_get_session_file` (580行目)
- **定義**: `app/agents/browser/session_manager.py:580-585`
- **利用箇所**: `_apply_saved_session` 内で使用
- **責務**: BrowserUseAgent版と同等の機能

### 3. settings.py 側の実装

#### `setup_routes` (202行目)
- **定義**: `app/agents/browser/settings.py:202-264`
- **利用箇所**: **未確認**（おそらく未使用）
- **責務**: BrowserUseAgent/SessionManager版と同等の機能（関数版）

#### `apply_saved_session` (277行目)
- **定義**: `app/agents/browser/settings.py:277-327`
- **利用箇所**: **未確認**（おそらく未使用）
- **責務**: BrowserUseAgent/SessionManager版と同等の機能（関数版）

#### `setup_init_scripts` (330行目)
- **定義**: `app/agents/browser/settings.py:330-450+`
- **利用箇所**: **未確認**（おそらく未使用）
- **責務**: BrowserUseAgent/SessionManager版と同等の機能（関数版）

#### `get_session_file` (267行目)
- **定義**: `app/agents/browser/settings.py:267-274`
- **利用箇所**: `apply_saved_session` 内で使用
- **責務**: BrowserUseAgent/SessionManager版と同等の機能（関数版）

---

## Step 2: 責務の最終方針

### 方針

- **SessionManager**: ブラウザセッション関連の**すべての実装**を担当
- **settings.py**: 純粋な設定値（定数）や軽いヘルパー関数のみ
- **BrowserUseAgent**: SessionManager の利用者に徹し、ブラウザ設定の実装詳細は一切持たない

### 決定事項（どの関数をどのファイルに残すか／削除するか）

#### ✅ SessionManager に残す（唯一の実装）
- `_build_context_options()` - プライベートメソッドとして保持
- `_setup_routes()` - プライベートメソッドとして保持
- `_setup_init_scripts()` - プライベートメソッドとして保持
- `_apply_saved_session()` - プライベートメソッドとして保持
- `_get_session_file()` - プライベートメソッドとして保持

#### ❌ BrowserUseAgent から削除
- `_build_context_options()` - **削除**（未使用のため）
- `_setup_routes()` - **削除**（未使用のため）
- `_apply_saved_session()` - **削除**（未使用のため）
- `_setup_init_scripts()` - **削除**（未使用のため）
- `_get_session_file()` - **削除**（未使用のため）

#### ⚠️ settings.py の整理
- `setup_routes()` - **削除**（SessionManager に統合済み）
- `apply_saved_session()` - **削除**（SessionManager に統合済み）
- `setup_init_scripts()` - **削除**（SessionManager に統合済み）
- `get_session_file()` - **削除**（SessionManager に統合済み）
- **ただし**: 定数（`EXTERNAL_BLOCKLIST_HOSTS`, `SESSION_DIR` 等）は残す

---

## 次のステップ

1. BrowserUseAgent から上記メソッドを削除
2. settings.py から上記関数を削除（定数は残す）
3. SessionManager の実装が正しく動作することを確認
4. テストの実行と確認

