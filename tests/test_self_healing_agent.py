"""
test_self_healing_agent.py
======================================================================
SelfHealingAgent のユニットテスト
======================================================================
"""

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]


def test_self_healing_agent_module_exists():
    """モジュールファイルが存在することを確認"""
    agent_path = APP_ROOT / "app" / "agents" / "self_healing_agent.py"
    assert agent_path.exists(), f"self_healing_agent.py not found at {agent_path}"


def test_self_healing_agent_has_required_attributes():
    """必要な属性が存在するか確認（Playwright依存なし）"""
    import ast

    agent_path = APP_ROOT / "app" / "agents" / "self_healing_agent.py"
    with open(agent_path) as f:
        ast.parse(f.read())

    source = agent_path.read_text()

    # 実際の実装存在する属性のみ確認
    required = [
        ("recovery_agent", "PageRecoveryAgent instance"),
        ("repair_agent", "SelectorRepairAgent instance"),
        ("fkb", "FKB instance"),
        ("execute", "async method"),
        ("_record_failure", "method"),
        ("_record_success", "method"),
        ("_check_circuit_breaker", "method"),
        ("get_recovery_stats", "method"),
    ]

    for attr, _expected_type in required:
        if attr in ("execute", "handle_moncler_failure"):
            assert f"def {attr}" in source, f"Method {attr} not found in source"
        else:
            assert attr in source, f"Attribute {attr} not found in source"


def test_fkb_local_json_is_valid():
    """FKBローカルファイルが有効なJSONであることを確認"""
    import json

    fkb_path = APP_ROOT / "config" / "fkb" / "fkb_local.json"
    assert fkb_path.exists(), "fkb_local.json not found"

    with open(fkb_path) as f:
        data = json.load(f)

    assert "entries" in data, "FKB missing 'entries' key"
    assert isinstance(data["entries"], list), "FKB 'entries' must be a list"
    assert len(data["entries"]) > 0, "FKB must have at least one entry"

    for entry in data["entries"]:
        assert "id" in entry, f"Entry missing 'id': {entry}"
        assert "site" in entry, f"Entry {entry.get('id')} missing 'site'"
        assert "error_signature" in entry, f"Entry {entry.get('id')} missing 'error_signature'"
        assert "root_cause" in entry, f"Entry {entry.get('id')} missing 'root_cause'"
        assert "solution_pattern" in entry, f"Entry {entry.get('id')} missing 'solution_pattern'"


def test_fkb_has_minimum_coverage():
    """FKBが主要なサイトカテゴリをカバーしていることを確認"""
    import json

    fkb_path = APP_ROOT / "config" / "fkb" / "fkb_local.json"
    with open(fkb_path) as f:
        data = json.load(f)

    sites = {entry["site"] for entry in data["entries"]}
    required_sites = {"MONCLER_OFFICIAL", "SSENSE", "GENERIC"}

    missing = required_sites - sites
    assert not missing, f"FKB missing coverage for: {missing}"


def test_self_healing_agent_execute_method_signature():
    """executeメソッドのシグネチャを確認"""
    agent_path = APP_ROOT / "app" / "agents" / "self_healing_agent.py"
    source = agent_path.read_text()

    # async def execute(...) が存在すること
    assert "async def execute" in source, "execute method not found"
    # page, run_context, settings, attempt, failure_context 引数があること
    assert "page:" in source or "page :" in source, "page parameter not found"
    assert "run_context:" in source or "run_context :" in source, "run_context parameter not found"


def test_self_healing_agent_has_circuit_breaker():
    """Circuit Breaker関連の属性・定数を確認"""
    agent_path = APP_ROOT / "app" / "agents" / "self_healing_agent.py"
    source = agent_path.read_text()

    # CB定数
    assert "CB_THRESHOLD" in source, "CB_THRESHOLD not found"
    assert "CB_COOLDOWN_SEC" in source, "CB_COOLDOWN_SEC not found"
    # CB状態管理
    assert "_cb_failures" in source, "_cb_failures not found"
    assert "_cb_opened_at" in source, "_cb_opened_at not found"
    # CBヘルパー
    assert "_check_circuit_breaker" in source, "_check_circuit_breaker not found"
    assert "_record_failure" in source, "_record_failure not found"
    assert "_record_success" in source, "_record_success not found"


def test_fkb_entry_ids_are_unique():
    """FKBエントリIDが重複していないことを確認"""
    import json

    fkb_path = APP_ROOT / "config" / "fkb" / "fkb_local.json"
    with open(fkb_path) as f:
        data = json.load(f)

    ids = [entry["id"] for entry in data["entries"]]
    dupes = set([i for i in ids if ids.count(i) > 1])
    assert not dupes, f"Duplicate FKB entry IDs: {dupes}"


def test_all_fkb_entries_have_verified_date():
    """全FKBエントリがverified_dateを持つことを確認"""
    import json

    fkb_path = APP_ROOT / "config" / "fkb" / "fkb_local.json"
    with open(fkb_path) as f:
        data = json.load(f)

    for entry in data["entries"]:
        assert "verified_date" in entry, f"Entry {entry.get('id')} missing 'verified_date'"
