#!/usr/bin/env python
"""
Qwen3-Coder + Ollama ile otonom coder agent (plan → apply + test fix + format/lint + review, v5).

Özellikler:
- Config'i okur
- Workspace hafızasını (.otonom_memory.md) prompt'a ekler
- PLAN turu (max_rounds >= 2 ise)
- CODER turu (dosya aksiyonları üretir)
- Dosyaları yazar/siler (dry-run destekli)
- Format komutu (ör. black) çalıştırır
- Lint komutu (ör. ruff) çalıştırır
- Test komutunu (ör. pytest) çalıştırır
- Test fail olursa:
    - Hata logunu modele verir
    - Yeni bir dosya aksiyonu turu ister
    - Dosyaları günceller
    - Testleri tekrar çalıştırır (max_test_fix_rounds kadar)
- Reviewer agent:
    - Değişiklikleri ve projeyi gözden geçirir, yorum üretir
- Plan + aksiyon + format/lint + test + review özetini string olarak döndürür
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from ollama import Client  # type: ignore


# Workspace hafıza dosyası ismi
MEMORY_FILENAME = ".otonom_memory.md"


# ---------------------------------------------------------------------------
# Config yapısı
# ---------------------------------------------------------------------------

@dataclass
class ModelEndpoint:
    host: str
    model: str


@dataclass
class GeneralConfig:
    default_workspace: str
    temperature: float
    max_rounds: int
    max_files_in_tree: int
    max_preview_bytes_per_file: int
    dry_run_default: bool
    run_tests_default: bool
    test_command: str
    max_test_fix_rounds: int
    test_timeout_seconds: int
    run_format_default: bool
    format_command: str
    run_lint_default: bool
    lint_command: str


@dataclass
class AppConfig:
    mode: str  # "cloud" veya "local"
    cloud: ModelEndpoint
    local: ModelEndpoint
    general: GeneralConfig


def load_config(config_path: Path) -> AppConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Ayar dosyası bulunamadı: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    mode = str(raw.get("mode", "cloud")).lower()

    cloud_raw = raw.get("cloud", {})
    local_raw = raw.get("local", {})
    general_raw = raw.get("general", {})

    cloud = ModelEndpoint(
        host=str(cloud_raw.get("host", "http://localhost:11434")),
        model=str(cloud_raw.get("model", "qwen3-coder:480b-cloud")),
    )
    local = ModelEndpoint(
        host=str(local_raw.get("host", "http://localhost:11434")),
        model=str(local_raw.get("model", "qwen2.5-coder:32b")),
    )
    general = GeneralConfig(
        default_workspace=str(general_raw.get("default_workspace", "./workspace")),
        temperature=float(general_raw.get("temperature", 0.1)),
        max_rounds=int(general_raw.get("max_rounds", 2)),
        max_files_in_tree=int(general_raw.get("max_files_in_tree", 150)),
        max_preview_bytes_per_file=int(
            general_raw.get("max_preview_bytes_per_file", 2000)
        ),
        dry_run_default=bool(general_raw.get("dry_run_default", False)),
        run_tests_default=bool(general_raw.get("run_tests_default", True)),
        test_command=str(general_raw.get("test_command", "pytest")),
        max_test_fix_rounds=int(general_raw.get("max_test_fix_rounds", 1)),
        test_timeout_seconds=int(general_raw.get("test_timeout_seconds", 300)),
        run_format_default=bool(general_raw.get("run_format_default", True)),
        format_command=str(general_raw.get("format_command", "black .")),
        run_lint_default=bool(general_raw.get("run_lint_default", False)),
        lint_command=str(general_raw.get("lint_command", "ruff .")),
    )

    return AppConfig(mode=mode, cloud=cloud, local=local, general=general)


def select_endpoint(cfg: AppConfig, override_mode: Optional[str] = None) -> ModelEndpoint:
    mode = override_mode.lower() if override_mode else cfg.mode.lower()
    if mode == "local":
        return cfg.local
    return cfg.cloud


def get_workspace_overview(workspace: Path, max_files: int) -> str:
    """
    Workspace içindeki dosya ağacını kısa bir metin olarak çıkar.
    Modeller için bağlam verir.
    """
    items: List[str] = []
    for p in workspace.rglob("*"):
        if p.is_file():
            rel = p.relative_to(workspace)
            items.append(str(rel))
            if len(items) >= max_files:
                break
    if not items:
        return "(empty workspace)"
    return "\n".join(items)


def load_workspace_memory(workspace: Path) -> str:
    """
    Workspace hafıza dosyasını (.otonom_memory.md) okur.
    """
    mem_file = workspace / MEMORY_FILENAME
    if mem_file.exists():
        try:
            return mem_file.read_text(encoding="utf-8")
        except Exception:
            return ""
    return ""


# ---------------------------------------------------------------------------
# PLANNER PROMPT
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """
You are a senior software architect AI.

Your job in this phase is ONLY to plan the code changes, not to write full file contents.

You will receive:
- Workspace absolute path
- Workspace file tree
- (Optionally) workspace memory (project notes, stack, style)
- A natural language task

You must output a single Markdown JSON code block describing the plan:

```json
{
  "summary": "short 1-3 sentence explanation of the plan",
  "files": [
    {
      "path": "relative/path/from/workspace.py",
      "role": "short role name (e.g. entrypoint, router, types, tests)",
      "description": "short description what will be implemented here"
    }
  ]
}
```

Rules:

* "path" MUST be relative to the workspace root (no .., no absolute paths).
* You MUST return valid JSON (no comments, no trailing commas).
* Outside the `json ... ` block you MUST NOT output anything.
"""

def build_planner_messages(task: str, workspace: Path, cfg: AppConfig) -> List[Dict[str, Any]]:
    file_tree = get_workspace_overview(workspace, cfg.general.max_files_in_tree)
    memory_text = load_workspace_memory(workspace)
    memory_part = ""
    if memory_text.strip():
        memory_part = "\n\nWorkspace memory:\n" + memory_text + "\n"

    user_content = (
        "Workspace absolute path:\n"
        f"{str(workspace.resolve())}\n\n"
        "Workspace file tree (relative paths, truncated):\n"
        f"{file_tree}\n"
        f"{memory_part}"
        "Task:\n"
        f"{task}"
    )
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# CODER PROMPT
# ---------------------------------------------------------------------------

CODER_SYSTEM_PROMPT = """
You are an autonomous coding agent similar to Claude Coder / OpenAI Codex.

You ONLY work inside a given workspace directory and you have FULL control over files:

* You can create, overwrite, and delete files.
* You must ALWAYS write full file contents (never patches, never diffs).
* Prefer the simplest, most robust solution.

### GOAL

The user will give you:

* A natural language task.
* The absolute path of the workspace directory.
* A file tree overview of the current workspace (relative paths).
* (Optionally) workspace memory (project notes, stack, style).
* (Optionally) a high-level file plan produced by a planner agent.
* (Optionally) test/format/lint failure info (stdout/stderr).

You must:

1. Understand the task and the planner's intent (if given).
2. If failure info is provided, focus on fixing the failure while preserving existing working code as much as possible.
3. Decide the minimal set of files to create/update/delete to complete the task.
4. Output a JSON array describing the file operations.

### FILE OPERATIONS FORMAT

You MUST respond with a single Markdown JSON code block:

```json
[
  {
    "path": "relative/path/from/workspace.py",
    "action": "write",
    "description": "short explanation for this file",
    "content": "FULL FILE CONTENT HERE"
  },
  {
    "path": "relative/old_file_to_delete.py",
    "action": "delete",
    "description": "why this file is removed"
  }
]
```

Rules:

* "path" MUST be relative to the workspace root (no .., no absolute paths).
* "action" is one of: "write", "delete".
* For "write", you MUST include the full final file content in "content".
* For "delete", do NOT include "content".
* You MUST return valid JSON (no comments, no trailing commas).
* Outside the `json ... ` block you MUST NOT output anything (no explanation, no markdown, nothing).

If the task is not clear, make reasonable assumptions and still produce a JSON file plan.
"""

def build_coder_messages(
    task: str,
    workspace: Path,
    cfg: AppConfig,
    planner_json_text: Optional[str],
    failure_info: Optional[str] = None,
) -> List[Dict[str, Any]]:
    file_tree = get_workspace_overview(workspace, cfg.general.max_files_in_tree)
    memory_text = load_workspace_memory(workspace)
    memory_part = ""
    if memory_text.strip():
        memory_part = "\n\nWorkspace memory:\n" + memory_text + "\n"

    planner_part = ""
    if planner_json_text:
        planner_part = (
            "\n\nHigh-level file plan (JSON from planner):\n"
            f"{planner_json_text}\n"
        )

    failure_part = ""
    if failure_info:
        failure_part = (
            "\n\nFailure info (format/lint/tests):\n"
            f"{failure_info}\n"
        )

    user_content = (
        "Workspace absolute path:\n"
        f"{str(workspace.resolve())}\n\n"
        "Workspace file tree (relative paths, truncated):\n"
        f"{file_tree}\n"
        f"{memory_part}"
        "Task:\n"
        f"{task}"
        f"{planner_part}"
        f"{failure_part}"
    )

    return [
        {"role": "system", "content": CODER_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# REVIEWER PROMPT
# ---------------------------------------------------------------------------

REVIEWER_SYSTEM_PROMPT = """
You are a senior code reviewer.

You will receive:

* Workspace absolute path
* Workspace file tree
* (Optionally) workspace memory (project notes, stack, style)
* A natural language task
* A summary of file changes (paths + actions)

Your job:

* Provide a concise code review focusing on:

  * Correctness risks
  * Maintainability
  * Test coverage hints
  * Simple refactor suggestions (if any)
* Do NOT output code, do NOT output JSON.
* Answer in natural language Markdown (bullet points welcome).
"""

def build_reviewer_messages(
    task: str,
    workspace: Path,
    cfg: AppConfig,
    all_actions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    file_tree = get_workspace_overview(workspace, cfg.general.max_files_in_tree)
    memory_text = load_workspace_memory(workspace)

    actions_lines: List[str] = []
    for a in all_actions:
        path = a.get("path", "?")
        op = a.get("action", "?")
        desc = a.get("description", "")
        tag = f"[{op}]" if op else "[?]"
        if desc:
            actions_lines.append(f"{tag} {path} → {desc}")
        else:
            actions_lines.append(f"{tag} {path}")
    actions_text = "\n".join(actions_lines) if actions_lines else "(no file changes)"

    memory_part = ""
    if memory_text.strip():
        memory_part = "\n\nWorkspace memory:\n" + memory_text + "\n"

    user_content = (
        "Workspace absolute path:\n"
        f"{str(workspace.resolve())}\n\n"
        "Workspace file tree (relative paths, truncated):\n"
        f"{file_tree}\n"
        f"{memory_part}"
        "Task:\n"
        f"{task}\n\n"
        "File changes:\n"
        f"{actions_text}"
    )

    return [
        {"role": "system", "content": REVIEWER_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# LLM çağrısı / JSON blok çıkarma
# ---------------------------------------------------------------------------

def call_llm(
    endpoint: ModelEndpoint,
    messages: List[Dict[str, Any]],
    temperature: float,
) -> str:
    client = Client(host=endpoint.host)
    response = client.chat(
        model=endpoint.model,
        messages=messages,
        options={"temperature": float(temperature)},
    )
    # Ollama python client dict döndürür: {"message": {"content": "..."}}
    return response["message"]["content"]  # type: ignore[index]

def extract_json_block(text: str) -> str:
    """
    Model çıktısından `json ... ` bloğunu çek.
    Eğer yoksa tüm metni JSON sanıp parse etmeyi deneriz.
    """
    start_token = "```json"
    end_token = "```"

    start_idx = text.find(start_token)
    if start_idx == -1:
        return text.strip()

    start_idx += len(start_token)
    end_idx = text.find(end_token, start_idx)
    if end_idx == -1:
        return text[start_idx:].strip()

    return text[start_idx:end_idx].strip()


# ---------------------------------------------------------------------------
# Dosya güvenliği ve uygulama
# ---------------------------------------------------------------------------

def ensure_path_in_workspace(workspace: Path, target: Path) -> None:
    """
    path traversal engelle: workspace dışına çıkmasın.
    """
    workspace_resolved = workspace.resolve()
    target_resolved = target.resolve()
    if workspace_resolved == target_resolved:
        return
    if workspace_resolved not in target_resolved.parents:
        raise ValueError(f"Workspace dışı path reddedildi: {target_resolved}")

def apply_file_actions(actions: List[Dict[str, Any]], workspace: Path) -> List[str]:
    """
    Dosya işlemlerini uygular ve kısa log satırları döner.
    """
    logs: List[str] = []

    for action in actions:
        path_str = action.get("path")
        op = action.get("action")
        if not path_str or not op:
            logs.append(f"[UYARI] Geçersiz action (path/action yok): {action}")
            continue

        target_path = workspace / path_str
        ensure_path_in_workspace(workspace, target_path)

        if op == "write":
            content = action.get("content", "")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            logs.append(f"[WRITE] {path_str} ({len(content)} byte)")
        elif op == "delete":
            if target_path.exists():
                target_path.unlink()
                logs.append(f"[DELETE] {path_str}")
            else:
                logs.append(f"[SKIP-DELETE] {path_str} (zaten yok)")
        else:
            logs.append(f"[UYARI] Bilinmeyen action türü: {op} ({path_str})")

    return logs


def parse_actions(json_text: str) -> List[Dict[str, Any]]:
    data = json.loads(json_text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "files" in data and isinstance(data["files"], list):
        return data["files"]
    raise ValueError("Beklenen format bir JSON listesi veya {'files': [...]} olmalı.")


# ---------------------------------------------------------------------------
# Komut çalıştırma (format, lint, test, manuel)
# ---------------------------------------------------------------------------

def run_tests(
    workspace: Path,
    command: str,
    timeout_seconds: int,
) -> Tuple[int, str, str]:
    """
    Test komutunu çalıştırır, (returncode, stdout, stderr) döner.
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return (
            124,
            e.stdout or "",
            (e.stderr or "") + f"\n\n[timeout] Komut {timeout_seconds} saniyede bitmedi.",
        )
    except Exception as e:
        return 125, "", f"Test komutu çalıştırılırken hata: {e}"

def summarize_test_result(
    rc: int,
    stdout: str,
    stderr: str,
    command: str,
) -> str:
    """
    Test sonucunu kısa bir metin olarak özetler.
    """
    max_len = 2000

    def cut(s: str) -> str:
        return s[:max_len] + ("..." if len(s) > max_len else "")

    if rc == 0:
        status_line = f"✅ Test komutu başarılı: `{command}` (rc=0)"
    else:
        status_line = f"❌ Test komutu başarısız: `{command}` (rc={rc})"

    out_part = cut(stdout)
    err_part = cut(stderr)

    return (
        status_line
        + "\n\n[stdout]\n"
        + (out_part or "(boş)")
        + "\n\n[stderr]\n"
        + (err_part or "(boş)")
    )


def run_generic_command(
    workspace: Path,
    command: str,
    timeout_seconds: int,
) -> Tuple[int, str, str]:
    """
    Genel amaçlı shell komutu çalıştır (format/lint/manual).
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return (
            124,
            e.stdout or "",
            (e.stderr or "") + f"\n\n[timeout] Komut {timeout_seconds} saniyede bitmedi.",
        )
    except Exception as e:
        return 125, "", f"Komut çalıştırılırken hata: {e}"

def summarize_generic_command_result(
    label: str,
    rc: int,
    stdout: str,
    stderr: str,
    command: str,
) -> str:
    """
    Genel amaçlı komut sonucu özetleyici (format/lint/manual).
    """
    max_len = 2000

    def cut(s: str) -> str:
        return s[:max_len] + ("..." if len(s) > max_len else "")

    if rc == 0:
        status_line = f"✅ {label} başarılı: `{command}` (rc=0)"
    else:
        status_line = f"❌ {label} başarısız: `{command}` (rc={rc})"

    out_part = cut(stdout)
    err_part = cut(stderr)

    return (
        status_line
        + "\n\n[stdout]\n"
        + (out_part or "(boş)")
        + "\n\n[stderr]\n"
        + (err_part or "(boş)")
    )


# ---------------------------------------------------------------------------
# Özetleyiciler
# ---------------------------------------------------------------------------

def summarize_plan(plan_obj: Optional[Dict[str, Any]], raw_json_text: str) -> str:
    if plan_obj is None:
        return "Plan JSON ayrıştırılamadı, coder doğrudan görev üzerinden karar verdi."

    summary = plan_obj.get("summary")
    files = plan_obj.get("files")

    lines: List[str] = []

    if summary:
        lines.append(summary)

    if isinstance(files, list) and files:
        lines.append("")
        lines.append("Dosya planı:")
        for f in files:
            path = f.get("path", "?")
            role = f.get("role", "")
            desc = f.get("description", "")
            tag = f"[{role}]" if role else ""
            if desc:
                lines.append(f"- {tag} {path} → {desc}")
            else:
                lines.append(f"- {tag} {path}")

    if not lines:
        short = raw_json_text[:400]
        if len(raw_json_text) > 400:
            short += "..."
        lines.append("Plan içeriği:")
        lines.append(short)

    return "\n".join(lines)


def summarize_actions(
    actions: List[Dict[str, Any]],
    logs: Optional[List[str]] = None,
    dry_run: bool = False,
    header_prefix: str = "",
) -> str:
    """
    Dosya aksiyonlarını insan-diliyle özetle.
    """
    if not actions:
        base = "Herhangi bir dosya değişikliği planlanmadı."
        if dry_run:
            base = "🟡 DRY-RUN (dosyalara dokunulmadı)\n\n" + base
        return (header_prefix + "\n" if header_prefix else "") + base

    write_count = 0
    delete_count = 0
    lines: List[str] = []

    for a in actions:
        path = a.get("path", "?")
        op = a.get("action", "?")
        desc = a.get("description", "")
        if op == "write":
            write_count += 1
            prefix = "YAZ"
        elif op == "delete":
            delete_count += 1
            prefix = "SİL"
        else:
            prefix = op.upper()
        if desc:
            lines.append(f"- [{prefix}] {path} → {desc}")
        else:
            lines.append(f"- [{prefix}] {path}")

    header = (
        f"{write_count} dosya yazılacak, {delete_count} dosya silinecek."
        if dry_run
        else f"{write_count} dosya yazıldı, {delete_count} dosya silindi."
    )

    result = header + "\n" + "\n".join(lines)

    if logs:
        result += "\n\nLog:\n" + "\n".join(logs)

    if dry_run:
        result = "🟡 DRY-RUN (dosyalara dokunulmadı)\n\n" + result
    else:
        result = "🟢 UYGULANDI\n\n" + result

    if header_prefix:
        result = header_prefix + "\n" + result

    return result


# ---------------------------------------------------------------------------
# Planner + Coder + Format/Lint + Test + Review ana fonksiyon
# ---------------------------------------------------------------------------

def run_agent_round(
    config: AppConfig,
    task: str,
    workspace: Path,
    mode_override: Optional[str] = None,
    dry_run: Optional[bool] = None,
    run_tests_flag: Optional[bool] = None,
    test_command_override: Optional[str] = None,
    run_format_flag: Optional[bool] = None,
    format_command_override: Optional[str] = None,
    run_lint_flag: Optional[bool] = None,
    lint_command_override: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Tek user görevi için tam pipeline:
    - İsteğe bağlı PLAN turu (max_rounds >= 2 ise)
    - CODER turu (dosya aksiyonlarını üretir)
    - İstenirse aksiyonları workspace'e uygular
    - İstenirse format ve lint komutlarını çalıştırır
    - İstenirse test komutunu çalıştırır
        - Test fail olursa, hata logunu modele verip yeni aksiyon isteyerek düzeltmeye çalışır
    - Reviewer agent ile code review çıktısı üretir
    - Tüm aksiyonları ve birleşik özet metnini döndürür
    """
    workspace.mkdir(parents=True, exist_ok=True)
    endpoint = select_endpoint(config, mode_override)

    # Dry-run & test/format/lint parametreleri
    dry_run_flag = config.general.dry_run_default if dry_run is None else dry_run

    run_tests_effective = (
        config.general.run_tests_default if run_tests_flag is None else run_tests_flag
    )
    test_command = (
        test_command_override
        if test_command_override is not None
        else config.general.test_command
    )

    run_format_effective = (
        config.general.run_format_default if run_format_flag is None else run_format_flag
    )
    format_command = (
        format_command_override
        if format_command_override is not None
        else config.general.format_command
    )

    run_lint_effective = (
        config.general.run_lint_default if run_lint_flag is None else run_lint_flag
    )
    lint_command = (
        lint_command_override
        if lint_command_override is not None
        else config.general.lint_command
    )

    all_actions: List[Dict[str, Any]] = []
    summary_chunks: List[str] = []

    # -------- PLAN TURU (isteğe bağlı) --------
    planner_plan_obj: Optional[Dict[str, Any]] = None
    planner_json_text_for_coder: Optional[str] = None
    planner_summary_text: Optional[str] = None

    use_planner = config.general.max_rounds >= 2

    if use_planner:
        planner_messages = build_planner_messages(task, workspace, config)
        planner_raw_output = call_llm(
            endpoint=endpoint,
            messages=planner_messages,
            temperature=config.general.temperature,
        )
        planner_json_text = extract_json_block(planner_raw_output)
        planner_json_text_for_coder = planner_json_text

        try:
            parsed_plan = json.loads(planner_json_text)
            if isinstance(parsed_plan, dict):
                planner_plan_obj = parsed_plan
            else:
                planner_plan_obj = None
        except Exception:
            planner_plan_obj = None

        planner_summary_text = summarize_plan(planner_plan_obj, planner_json_text)
        summary_chunks.append("📋 Plan\n" + planner_summary_text)

    # -------- CODER TURU (ilk tur) --------
    coder_messages = build_coder_messages(
        task=task,
        workspace=workspace,
        cfg=config,
        planner_json_text=planner_json_text_for_coder,
        failure_info=None,
    )

    coder_raw_output = call_llm(
        endpoint=endpoint,
        messages=coder_messages,
        temperature=config.general.temperature,
    )
    coder_json_text = extract_json_block(coder_raw_output)

    try:
        actions = parse_actions(coder_json_text)
    except Exception as e:
        err_summary = (
            "❌ Model çıktısı geçersiz JSON gibi görünüyor, dosyalara dokunulmadı.\n\n"
            f"Hata: {e}\n\n"
            "Ham çıktı (kısaltılmış):\n"
            f"{coder_raw_output[:2000]}{'...' if len(coder_raw_output) > 2000 else ''}"
        )
        summary_chunks.append("📦 Dosya Aksiyonları\n" + err_summary)
        final_summary = "\n\n".join(summary_chunks)
        return [], final_summary

    all_actions.extend(actions)

    logs: List[str] = []
    if not dry_run_flag:
        try:
            logs = apply_file_actions(actions, workspace)
        except Exception as e:
            err_summary2 = (
                "❌ Dosya işlemlerinde hata oluştu.\n\n"
                f"Hata: {e}"
            )
            summary_chunks.append("📦 Dosya Aksiyonları\n" + err_summary2)
            final_summary2 = "\n\n".join(summary_chunks)
            return all_actions, final_summary2

    actions_summary = summarize_actions(
        actions, logs, dry_run=dry_run_flag, header_prefix="📦 Dosya Aksiyonları (İlk Tur)"
    )
    summary_chunks.append(actions_summary)

    # -------- FORMAT & LINT (dry-run değilse) --------
    combined_failure_info = ""

    if not dry_run_flag:
        # Format
        if run_format_effective and format_command.strip():
            rc_f, out_f, err_f = run_generic_command(
                workspace=workspace,
                command=format_command,
                timeout_seconds=config.general.test_timeout_seconds,
            )
            fmt_summary = summarize_generic_command_result(
                "Format komutu", rc_f, out_f, err_f, format_command
            )
            summary_chunks.append("🎨 Format Sonucu\n" + fmt_summary)
            if rc_f != 0:
                combined_failure_info += "\n\n[FORMAT FAILURE]\n" + fmt_summary

        # Lint
        if run_lint_effective and lint_command.strip():
            rc_l, out_l, err_l = run_generic_command(
                workspace=workspace,
                command=lint_command,
                timeout_seconds=config.general.test_timeout_seconds,
            )
            lint_summary = summarize_generic_command_result(
                "Lint komutu", rc_l, out_l, err_l, lint_command
            )
            summary_chunks.append("🔍 Lint Sonucu\n" + lint_summary)
            if rc_l != 0:
                combined_failure_info += "\n\n[LINT FAILURE]\n" + lint_summary

    # -------- TESTLER (isteğe bağlı) --------
    last_test_summary = ""
    if not dry_run_flag and run_tests_effective:
        rc, out, err = run_tests(
            workspace=workspace,
            command=test_command,
            timeout_seconds=config.general.test_timeout_seconds,
        )
        test_summary = summarize_test_result(rc, out, err, test_command)
        last_test_summary = test_summary
        summary_chunks.append("🧪 Test Sonucu (İlk Tur)\n" + test_summary)

        # Başarılıysa bitti
        if rc == 0 or config.general.max_test_fix_rounds <= 0:
            # Reviewer çağır
            if all_actions:
                review_messages = build_reviewer_messages(
                    task=task,
                    workspace=workspace,
                    cfg=config,
                    all_actions=all_actions,
                )
                review_raw = call_llm(
                    endpoint=endpoint,
                    messages=review_messages,
                    temperature=config.general.temperature,
                )
                summary_chunks.append("👀 Code Review\n" + review_raw)

            final_summary_ok = "\n\n".join(summary_chunks)
            return all_actions, final_summary_ok

        # -------- TEST FIX TUR(LAR)I --------
        remaining_fixes = config.general.max_test_fix_rounds
        failure_info_for_llm = last_test_summary
        if combined_failure_info.strip():
            failure_info_for_llm += combined_failure_info

        fix_round_index = 1
        while remaining_fixes > 0:
            remaining_fixes -= 1

            fix_messages = build_coder_messages(
                task=task,
                workspace=workspace,
                cfg=config,
                planner_json_text=planner_json_text_for_coder,
                failure_info=failure_info_for_llm,
            )

            fix_raw_output = call_llm(
                endpoint=endpoint,
                messages=fix_messages,
                temperature=config.general.temperature,
            )
            fix_json_text = extract_json_block(fix_raw_output)

            try:
                fix_actions = parse_actions(fix_json_text)
            except Exception as e:
                fix_err_summary = (
                    f"❌ Test düzeltme turu #{fix_round_index} için model çıktısı geçersiz JSON.\n\n"
                    f"Hata: {e}\n\n"
                    "Ham çıktı (kısaltılmış):\n"
                    f"{fix_raw_output[:2000]}{'...' if len(fix_raw_output) > 2000 else ''}"
                )
                summary_chunks.append(f"🛠 Test Düzeltme Turu #{fix_round_index}\n" + fix_err_summary)
                break

            all_actions.extend(fix_actions)

            fix_logs: List[str] = []
            try:
                fix_logs = apply_file_actions(fix_actions, workspace)
            except Exception as e:
                fix_err2 = (
                    f"❌ Test düzeltme turu #{fix_round_index} dosya işlemlerinde hata.\n\n"
                    f"Hata: {e}"
                )
                summary_chunks.append(f"🛠 Test Düzeltme Turu #{fix_round_index}\n" + fix_err2)
                break

            fix_actions_summary = summarize_actions(
                fix_actions,
                fix_logs,
                dry_run=False,
                header_prefix=f"🛠 Test Düzeltme Turu #{fix_round_index} - Dosya Aksiyonları",
            )
            summary_chunks.append(fix_actions_summary)

            # Tekrar test
            rc2, out2, err2 = run_tests(
                workspace=workspace,
                command=test_command,
                timeout_seconds=config.general.test_timeout_seconds,
            )
            test_summary2 = summarize_test_result(rc2, out2, err2, test_command)
            last_test_summary = test_summary2
            summary_chunks.append(
                f"🧪 Test Sonucu (Düzeltme Turu #{fix_round_index})\n" + test_summary2
            )

            if rc2 == 0:
                break

            failure_info_for_llm = test_summary2
            fix_round_index += 1

    # -------- Reviewer (dry-run değilse) --------
    if not dry_run_flag and all_actions:
        review_messages = build_reviewer_messages(
            task=task,
            workspace=workspace,
            cfg=config,
            all_actions=all_actions,
        )
        review_raw = call_llm(
            endpoint=endpoint,
            messages=review_messages,
            temperature=config.general.temperature,
        )
        summary_chunks.append("👀 Code Review\n" + review_raw)

    final_summary = "\n\n".join(summary_chunks)
    return all_actions, final_summary
