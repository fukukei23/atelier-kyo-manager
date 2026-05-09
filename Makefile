# ==== 共通設定 ========================================================

VENV_DIR ?= venv
PYTHON   ?= $(VENV_DIR)/bin/python
PIP      ?= $(VENV_DIR)/bin/pip

REQ_MAIN ?= requirements.txt
REQ_DEV  ?= requirements-dev.txt

# プロジェクト固有の設定・ターゲットは Makefile.local で上書き可能
-include Makefile.local

# ==== ヘルプ ==========================================================

.PHONY: help
help:
	@echo "Common Makefile (NexusCore / atelier-kyo-manager 共通)"
	@echo ""
	@echo "  make venv          - Python 仮想環境(venv)を作成"
	@echo "  make install       - requirements.txt をインストール"
	@echo "  make install-dev   - requirements + requirements-dev をインストール"
	@echo "  make test          - pytest 実行"
	@echo "  make lint          - ruff / flake8 等があれば実行"
	@echo "  make format        - black 等でフォーマット（あれば）"
	@echo "  make clean-pyc     - *.pyc, __pycache__ を削除"
	@echo "  make info          - Python / pip / プロジェクト情報表示"
	@echo ""
	@echo "  ※ プロジェクト固有の run ターゲットは Makefile.local で定義してください。"

# ==== venv / install ==================================================

$(VENV_DIR)/bin/python:
	@echo ">>> Create venv in $(VENV_DIR)"
	python3 -m venv $(VENV_DIR)

.PHONY: venv
venv: $(VENV_DIR)/bin/python

.PHONY: install
install: venv
	@echo ">>> Install main requirements..."
	@if [ -f "$(REQ_MAIN)" ]; then \
		$(PIP) install --upgrade pip && \
		$(PIP) install -r $(REQ_MAIN); \
	else \
		echo "[WARN] $(REQ_MAIN) が存在しません。"; \
	fi

.PHONY: install-dev
install-dev: install
	@echo ">>> Install dev requirements..."
	@if [ -f "$(REQ_DEV)" ]; then \
		$(PIP) install -r $(REQ_DEV); \
	else \
		echo "[INFO] $(REQ_DEV) は存在しません。スキップします。"; \
	fi

# ==== test / lint / format ============================================

.PHONY: test
test: venv
	@echo ">>> Run pytest..."
	@if [ -d "tests" ]; then \
		$(PYTHON) -m pytest -q; \
	else \
		echo "[WARN] tests ディレクトリがありません。"; \
	fi

.PHONY: lint
lint: venv
	@echo ">>> Run ruff check..."
	@if command -v $(VENV_DIR)/bin/ruff >/dev/null 2>&1; then \
		$(VENV_DIR)/bin/ruff check app/ tests/ scripts/ dev_tools/; \
	else \
		echo "[INFO] ruff が見つかりません。make install-dev を実行してください。"; \
	fi

.PHONY: format
format: venv
	@echo ">>> Run ruff format..."
	@if command -v $(VENV_DIR)/bin/ruff >/dev/null 2>&1; then \
		$(VENV_DIR)/bin/ruff format app/ tests/ scripts/ dev_tools/; \
	else \
		echo "[INFO] ruff が見つかりません。make install-dev を実行してください。"; \
	fi

# ==== clean ===========================================================

.PHONY: clean-pyc
clean-pyc:
	@echo ">>> Remove __pycache__ and *.pyc ..."
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	find . -type f -name "*.py[co]" -delete || true

# ==== info ============================================================

.PHONY: info
info: venv
	@echo ">>> Environment info"
	@echo "VENV_DIR = $(VENV_DIR)"
	@$(PYTHON) --version
	@$(PIP) list | head

