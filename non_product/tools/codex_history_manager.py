#!/usr/bin/env python3
"""
Codex history synchronizer.

This utility copies Codex CLI session logs together with Git diffs and any
referenced artifacts (images / diff files) into the repository-local
``codex_history`` folder.  It is designed to keep an auditable trail of every
Codex run without manually exporting logs from the CLI UI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_SESSION_ROOT = Path.home() / ".codex" / "sessions"
DEFAULT_HISTORY_ROOT = Path(__file__).resolve().parents[1] / "codex_history"
DEFAULT_SETTLE_SECONDS = 20.0
STATE_FILENAME = ".codex_history_state.json"

ARTIFACT_PATTERN = re.compile(
    r"((?:[A-Za-z]:\\|/)[^\"'\s]+?\.(?:png|jpg|jpeg|gif|webp|diff|patch))",
    re.IGNORECASE,
)
WINDOWS_PATH_PREFIX = re.compile(r"^[A-Za-z]:\\")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
DIFF_EXTENSIONS = {".diff", ".patch"}


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(value: str, fallback: str = "session") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "", value or "")
    return cleaned or fallback


def _parse_timestamp(value: Optional[str], default: Optional[datetime] = None) -> datetime:
    if not value:
        return default or datetime.now(timezone.utc)
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return default or datetime.now(timezone.utc)


class CodexHistoryManager:
    def __init__(
        self,
        project_root: Path,
        sessions_dir: Path = DEFAULT_SESSION_ROOT,
        history_root: Path = DEFAULT_HISTORY_ROOT,
        settle_seconds: float = DEFAULT_SETTLE_SECONDS,
        dry_run: bool = False,
    ) -> None:
        self.project_root = project_root
        self.sessions_dir = sessions_dir
        self.history_root = history_root
        self.settle_seconds = settle_seconds
        self.dry_run = dry_run

        self.run_logs_dir = history_root / "run_logs"
        self.git_diffs_dir = history_root / "git_diffs"
        self.metadata_dir = history_root / "metadata"
        self.artifact_images_dir = history_root / "artifacts" / "images"
        self.artifact_diffs_dir = history_root / "artifacts" / "diffs"
        self.state_path = history_root / STATE_FILENAME

        for directory in (
            self.run_logs_dir,
            self.git_diffs_dir,
            self.metadata_dir,
            self.artifact_images_dir,
            self.artifact_diffs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.state: Dict[str, Dict[str, str]] = self._load_state()

    # ------------------------------------------------------------------ public
    def sync_once(self) -> int:
        if not self.sessions_dir.exists():
            logging.warning("Codex sessions directory not found: %s", self.sessions_dir)
            return 0

        now = time.time()
        processed = 0
        for session_file in sorted(self.sessions_dir.rglob("*.jsonl")):
            if not session_file.is_file():
                continue
            try:
                rel_key = str(session_file.relative_to(self.sessions_dir))
            except ValueError:
                rel_key = str(session_file)

            mtime = session_file.stat().st_mtime
            if now - mtime < self.settle_seconds:
                logging.debug("Skipping %s (waiting for file to settle)", rel_key)
                continue

            file_hash = self._hash_file(session_file)
            prev_entry = self.state.get(rel_key)
            if prev_entry and prev_entry.get("hash") == file_hash:
                continue

            try:
                manifest_entry = self._process_session(session_file, rel_key, file_hash)
            except Exception as exc:
                logging.exception("Failed to process Codex session %s: %s", rel_key, exc)
                continue

            if manifest_entry:
                self.state[rel_key] = manifest_entry
                processed += 1

        if processed and not self.dry_run:
            self._save_state()
        return processed

    # ----------------------------------------------------------------- helpers
    def _process_session(
        self,
        session_file: Path,
        rel_key: str,
        file_hash: str,
    ) -> Optional[Dict[str, str]]:
        meta = self._extract_session_meta(session_file)
        timestamp = _parse_timestamp(
            meta.get("started_at"),
            default=datetime.fromtimestamp(session_file.stat().st_mtime, tz=timezone.utc),
        )
        target_stem = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{_slugify(meta.get('session_id', 'session'))}"
        logging.info("Capturing Codex session %s -> %s", rel_key, target_stem)

        run_log_path = self.run_logs_dir / f"{target_stem}.jsonl"
        if not self.dry_run:
            shutil.copy2(session_file, run_log_path)

        artifact_sources = self._extract_artifact_paths(session_file)
        copied_artifacts = self._copy_artifacts(target_stem, artifact_sources)

        git_state = self._capture_git_state()
        diff_path = self._write_git_diff(target_stem)

        metadata = {
            "session_id": meta.get("session_id"),
            "started_at": meta.get("started_at"),
            "last_event_ts": meta.get("last_event_ts"),
            "codex_originator": meta.get("originator"),
            "codex_cli_version": meta.get("cli_version"),
            "codex_source": meta.get("source"),
            "codex_cwd": meta.get("cwd"),
            "codex_git": meta.get("git"),
            "captured_at": _iso_now(),
            "project_root": str(self.project_root),
            "run_log": self._rel_or_abs(run_log_path),
            "git_diff": self._rel_or_abs(diff_path) if diff_path else None,
            "artifact_files": [self._rel_or_abs(p) for p in copied_artifacts],
            "git_branch": git_state.get("branch"),
            "git_head": git_state.get("head"),
            "git_status": git_state.get("status"),
            "git_error": git_state.get("error"),
            "session_source_file": str(session_file),
            "session_hash": file_hash,
        }

        metadata_path = self.metadata_dir / f"{target_stem}.json"
        if not self.dry_run:
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest_entry = {
            "hash": file_hash,
            "run_log": self._rel_or_abs(run_log_path),
            "metadata": self._rel_or_abs(metadata_path),
        }
        if diff_path:
            manifest_entry["git_diff"] = self._rel_or_abs(diff_path)
        if copied_artifacts:
            manifest_entry["artifacts"] = [self._rel_or_abs(p) for p in copied_artifacts]
        return manifest_entry

    def _extract_session_meta(self, session_file: Path) -> Dict[str, Optional[str]]:
        result: Dict[str, Optional[str]] = {
            "session_id": session_file.stem,
            "started_at": None,
            "cwd": None,
            "originator": None,
            "cli_version": None,
            "source": None,
            "git": None,
            "last_event_ts": None,
        }
        with session_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_ts = event.get("timestamp")
                if event_ts:
                    result["last_event_ts"] = event_ts
                if event.get("type") == "session_meta":
                    payload = event.get("payload") or {}
                    result["session_id"] = payload.get("id") or result["session_id"]
                    result["started_at"] = payload.get("timestamp") or result["started_at"]
                    result["cwd"] = payload.get("cwd")
                    result["originator"] = payload.get("originator")
                    result["cli_version"] = payload.get("cli_version")
                    result["source"] = payload.get("source")
                    result["git"] = payload.get("git")
                    break
        return result

    def _extract_artifact_paths(self, session_file: Path) -> List[Path]:
        artifacts: List[Path] = []
        with session_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for path_string in self._walk_strings(event):
                    normalized = self._normalize_path(path_string)
                    if normalized and normalized.exists():
                        artifacts.append(normalized)
                for match in ARTIFACT_PATTERN.finditer(raw_line):
                    normalized = self._normalize_path(match.group(1))
                    if normalized and normalized.exists():
                        artifacts.append(normalized)
        unique_artifacts: List[Path] = []
        seen: set[str] = set()
        for path in artifacts:
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                unique_artifacts.append(path)
        return unique_artifacts

    def _walk_strings(self, node: object) -> Iterable[str]:
        if isinstance(node, dict):
            for value in node.values():
                yield from self._walk_strings(value)
        elif isinstance(node, list):
            for value in node:
                yield from self._walk_strings(value)
        elif isinstance(node, str):
            yield node

    def _normalize_path(self, raw_value: str) -> Optional[Path]:
        candidate = raw_value.strip().strip('"').strip("'")
        if not candidate:
            return None
        candidate = candidate.replace("file://", "")
        candidate = candidate.replace("\\\\", "\\")

        potential: List[Path] = []
        if WINDOWS_PATH_PREFIX.match(candidate):
            drive = candidate[0].lower()
            remainder = candidate[2:].lstrip("\\/")
            mapped = Path("/mnt") / drive / Path(remainder.replace("\\", "/"))
            potential.append(mapped)
        else:
            potential.append(Path(candidate).expanduser())

        if not potential[-1].is_absolute():
            potential.append((self.project_root / potential[-1]).resolve())

        for path in potential:
            if path.exists():
                return path
        return None

    def _copy_artifacts(self, stem: str, sources: List[Path]) -> List[Path]:
        copied: List[Path] = []
        for source in sources:
            suffix = source.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                base_dir = self.artifact_images_dir / stem
            elif suffix in DIFF_EXTENSIONS:
                base_dir = self.artifact_diffs_dir / stem
            else:
                continue
            base_dir.mkdir(parents=True, exist_ok=True)
            target = base_dir / source.name
            if not self.dry_run:
                shutil.copy2(source, target)
            copied.append(target)
        return copied

    def _capture_git_state(self) -> Dict[str, Optional[str]]:
        state: Dict[str, Optional[str]] = {"branch": None, "head": None, "status": None, "error": None}
        try:
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            status = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            state["branch"] = branch.stdout.strip() or None
            state["head"] = head.stdout.strip() or None
            state["status"] = status.stdout.strip() or None
            for result in (branch, head, status):
                if result.returncode != 0 and result.stderr:
                    state["error"] = (state["error"] or "") + result.stderr.strip()
        except FileNotFoundError:
            state["error"] = "git executable not found"
        return state

    def _write_git_diff(self, stem: str) -> Optional[Path]:
        try:
            diff = subprocess.run(
                ["git", "diff"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            logging.warning("git executable not found; skipping diff export")
            return None

        diff_text = diff.stdout
        if not diff_text.strip():
            diff_text = "# No working tree changes captured by codex_history_manager.\n"
        target = self.git_diffs_dir / f"{stem}.diff"
        if not self.dry_run:
            target.write_text(diff_text, encoding="utf-8")
        return target

    def _hash_file(self, session_file: Path) -> str:
        digest = sha256()
        with session_file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_state(self) -> Dict[str, Dict[str, str]]:
        if not self.state_path.exists():
            return {}
        try:
            content = self.state_path.read_text(encoding="utf-8")
            return json.loads(content)
        except (json.JSONDecodeError, OSError):
            logging.warning("State file %s is corrupt or unreadable; recreating.", self.state_path)
            backup = self.state_path.with_suffix(".bak")
            try:
                shutil.copy2(self.state_path, backup)
            except OSError:
                pass
            return {}

    def _save_state(self) -> None:
        state_json = json.dumps(self.state, ensure_ascii=False, indent=2)
        self.state_path.write_text(state_json, encoding="utf-8")

    def _rel_or_abs(self, path: Optional[Path]) -> Optional[str]:
        if not path:
            return None
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync Codex CLI session logs into codex_history.")
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=DEFAULT_SESSION_ROOT,
        help=f"Override Codex session directory (default: {DEFAULT_SESSION_ROOT})",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_ROOT,
        help=f"Override target history directory (default: {DEFAULT_HISTORY_ROOT})",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=DEFAULT_SETTLE_SECONDS,
        help="Minimum age (in seconds) before a session file is considered stable.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll the sessions directory indefinitely.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Polling interval in seconds when --watch is enabled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended actions without writing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    project_root = Path(__file__).resolve().parents[1]
    manager = CodexHistoryManager(
        project_root=project_root,
        sessions_dir=args.sessions_dir,
        history_root=args.history_dir,
        settle_seconds=args.settle_seconds,
        dry_run=args.dry_run,
    )

    try:
        if args.watch:
            logging.info("Watching %s for new Codex sessions...", manager.sessions_dir)
            while True:
                synced = manager.sync_once()
                if synced:
                    logging.info("Captured %s new Codex session(s).", synced)
                time.sleep(max(args.interval, 5.0))
        else:
            synced = manager.sync_once()
            logging.info("Capture finished. %s session(s) processed.", synced)
    except KeyboardInterrupt:
        logging.info("Interrupted. Exiting Codex history manager.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
