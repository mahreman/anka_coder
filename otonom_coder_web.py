#!/usr/bin/env python
"""
Streamlit tabanlı web arayüzü
(plan → apply + test fix + format/lint + review + auto pip install, v6).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from otonom_coder_agent import (
    AppConfig,
    load_config,
    run_agent_round,
    run_generic_command,
    summarize_generic_command_result,
    MEMORY_FILENAME,
)


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"]: List[Dict[str, Any]] = []
    if "config" not in st.session_state:
        st.session_state["config"]: Optional[AppConfig] = None
    if "last_actions" not in st.session_state:
        st.session_state["last_actions"]: List[Dict[str, Any]] = []
    if "current_mode" not in st.session_state:
        st.session_state["current_mode"] = "cloud"
    if "workspace" not in st.session_state:
        st.session_state["workspace"] = "./workspace"
    if "config_path" not in st.session_state:
        st.session_state["config_path"] = "otonom_coder.config.yaml"
    if "dry_run" not in st.session_state:
        st.session_state["dry_run"] = False
    if "run_tests" not in st.session_state:
        st.session_state["run_tests"] = True
    if "test_command" not in st.session_state:
        st.session_state["test_command"] = "pytest"
    if "run_format" not in st.session_state:
        st.session_state["run_format"] = True
    if "format_command" not in st.session_state:
        st.session_state["format_command"] = "black ."
    if "run_lint" not in st.session_state:
        st.session_state["run_lint"] = False
    if "lint_command" not in st.session_state:
        st.session_state["lint_command"] = "ruff ."
    if "auto_install" not in st.session_state:
        st.session_state["auto_install"] = True
    if "pip_install_command" not in st.session_state:
        st.session_state["pip_install_command"] = "pip install {package}"
    if "selected_file" not in st.session_state:
        st.session_state["selected_file"] = ""
    if "workspace_memory_text" not in st.session_state:
        st.session_state["workspace_memory_text"] = ""
    if "manual_command" not in st.session_state:
        st.session_state["manual_command"] = "pytest"
    if "manual_command_history" not in st.session_state:
        st.session_state["manual_command_history"]: List[str] = []


def load_config_safe(config_path: str) -> AppConfig:
    cfg = load_config(Path(config_path))
    return cfg


def list_workspace_files(workspace: Path, max_files: int = 300) -> List[str]:
    if not workspace.exists():
        return []
    items: List[str] = []
    for p in workspace.rglob("*"):
        if p.is_file():
            rel = p.relative_to(workspace)
            items.append(str(rel))
            if len(items) >= max_files:
                break
    return sorted(items)


def load_workspace_memory_file(workspace: Path) -> str:
    mem_file = workspace / MEMORY_FILENAME
    if mem_file.exists():
        try:
            return mem_file.read_text(encoding="utf-8")
        except Exception as e:
            return f"# Hafıza dosyası okunamadı: {e}"
    return ""


def save_workspace_memory_file(workspace: Path, content: str) -> Optional[str]:
    try:
        mem_file = workspace / MEMORY_FILENAME
        mem_file.parent.mkdir(parents=True, exist_ok=True)
        mem_file.write_text(content, encoding="utf-8")
        return None
    except Exception as e:
        return str(e)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config("Otonom Coder", layout="wide")
    init_session_state()

    st.title("🧠 Otonom Coder (Qwen3-Coder + Ollama)")

    # Sidebar: config, mod, workspace, dry-run, format/lint/test/auto-install
    with st.sidebar:
        st.header("⚙️ Ayarlar")

        config_path = st.text_input(
            "Config dosyası",
            value=st.session_state["config_path"],
            help="otonom_coder.config.yaml yolunu gir.",
        )
        reload_btn = st.button("Config Yükle / Yenile")

        if reload_btn or st.session_state["config"] is None or config_path != st.session_state["config_path"]:
            try:
                cfg = load_config_safe(config_path)
                st.session_state["config"] = cfg
                st.session_state["config_path"] = config_path
                st.session_state["dry_run"] = cfg.general.dry_run_default
                st.session_state["run_tests"] = cfg.general.run_tests_default
                st.session_state["test_command"] = cfg.general.test_command
                st.session_state["run_format"] = cfg.general.run_format_default
                st.session_state["format_command"] = cfg.general.format_command
                st.session_state["run_lint"] = cfg.general.run_lint_default
                st.session_state["lint_command"] = cfg.general.lint_command
                st.session_state["auto_install"] = cfg.general.auto_install_missing_packages_default
                st.session_state["pip_install_command"] = cfg.general.pip_install_command
                st.success("Config yüklendi.")
            except Exception as e:
                st.session_state["config"] = None
                st.error(f"Config yüklenemedi: {e}")

        mode = st.radio(
            "Çalışma modu",
            options=["cloud", "local"],
            index=0 if st.session_state["current_mode"] == "cloud" else 1,
            horizontal=True,
        )
        st.session_state["current_mode"] = mode

        workspace = st.text_input(
            "Workspace klasörü",
            value=st.session_state["workspace"],
            help="Görevlerde kullanılacak çalışma klasörü.",
        )
        st.session_state["workspace"] = workspace

        if st.button("Workspace oluştur"):
            try:
                Path(workspace).mkdir(parents=True, exist_ok=True)
                st.success(f"Workspace hazır: {Path(workspace).resolve()}")
            except Exception as e:
                st.error(f"Workspace oluşturulamadı: {e}")

        dry_run_flag = st.checkbox(
            "Dry-run (dosyalara dokunma, sadece plan/aksiyon üret)",
            value=st.session_state["dry_run"],
        )
        st.session_state["dry_run"] = dry_run_flag

        st.markdown("### Test / Format / Lint")

        run_tests_flag = st.checkbox(
            "Testleri çalıştır (örn. pytest)",
            value=st.session_state["run_tests"],
        )
        st.session_state["run_tests"] = run_tests_flag

        test_cmd = st.text_input(
            "Test komutu",
            value=st.session_state["test_command"],
            help="Örn: pytest, python -m pytest, cargo test, npm test",
        )
        st.session_state["test_command"] = test_cmd

        run_format_flag = st.checkbox(
            "Format komutunu çalıştır (örn. black)",
            value=st.session_state["run_format"],
        )
        st.session_state["run_format"] = run_format_flag

        fmt_cmd = st.text_input(
            "Format komutu",
            value=st.session_state["format_command"],
            help="Örn: black ., yapıya göre değiştir.",
        )
        st.session_state["format_command"] = fmt_cmd

        run_lint_flag = st.checkbox(
            "Lint komutunu çalıştır (örn. ruff)",
            value=st.session_state["run_lint"],
        )
        st.session_state["run_lint"] = run_lint_flag

        lint_cmd = st.text_input(
            "Lint komutu",
            value=st.session_state["lint_command"],
            help="Örn: ruff ., mypy ., eslint .",
        )
        st.session_state["lint_command"] = lint_cmd

        st.markdown("### Auto pip install")

        auto_install_flag = st.checkbox(
            "ModuleNotFoundError için eksik paketi otomatik `pip install` et",
            value=st.session_state["auto_install"],
        )
        st.session_state["auto_install"] = auto_install_flag

        pip_cmd_template = st.text_input(
            "pip install komutu",
            value=st.session_state["pip_install_command"],
            help="Örn: pip install {package}  (package yeri otomatik doldurulacak)",
        )
        st.session_state["pip_install_command"] = pip_cmd_template

        st.markdown("---")
        st.caption("Görevleri aşağıdaki sohbet kutusundan yaz.")

    cfg: Optional[AppConfig] = st.session_state["config"]

    # Üst bilgi + explorer
    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader("Durum")
        if cfg is None:
            st.warning("Config henüz yüklü değil.")
        else:
            active_mode = st.session_state["current_mode"]
            model_name = (
                cfg.cloud.model if active_mode == "cloud" else cfg.local.model
            )
            rounds = cfg.general.max_rounds
            planner_on = rounds >= 2

            st.write(f"**Mod:** `{active_mode}`")
            st.write(f"**Model:** `{model_name}`")
            st.write(f"**Workspace:** `{Path(st.session_state['workspace']).resolve()}`")
            st.write(f"**Dry-run:** `{st.session_state['dry_run']}`")
            st.write(f"**Planlama turları (max_rounds):** `{rounds}`")
            st.write(f"**Planner aktif mi?** `{planner_on}`")
            st.write(f"**Testleri çalıştır:** `{st.session_state['run_tests']}`")
            st.write(f"**Test komutu:** `{st.session_state['test_command']}`")
            st.write(f"**Format komutu:** `{st.session_state['format_command']}` (aktif: {st.session_state['run_format']})")
            st.write(f"**Lint komutu:** `{st.session_state['lint_command']}` (aktif: {st.session_state['run_lint']})")
            st.write(f"**Auto pip install:** `{st.session_state['auto_install']}`")
            st.write(f"**pip komut şablonu:** `{st.session_state['pip_install_command']}`")
            st.write(
                f"**Max test fix turu:** `{cfg.general.max_test_fix_rounds}`"
            )

        st.markdown(
            "_ModuleNotFoundError gördüğünde, auto-install açıksa eksik paketi pip ile yükleyip testleri tekrar dener._"
        )
        )

    with col2:
        st.subheader("📁 Workspace Explorer")

        workspace_path = Path(st.session_state["workspace"])
        files = list_workspace_files(workspace_path)

        if not files:
            st.info("Workspace içinde henüz dosya yok.")
        else:
            selected = st.selectbox(
                "Bir dosya seç ve içeriğini gör:",
                options=["(seç)"] + files,
                index=0,
            )
            if selected != "(seç)":
                st.session_state["selected_file"] = selected
                target = workspace_path / selected
                try:
                    content = target.read_text(encoding="utf-8")
                except Exception as e:
                    content = f"Dosya okunamadı: {e}"
                st.code(content, language="python")

    st.markdown("---")

    # Workspace Memory
    st.subheader("🧠 Workspace Memory (.otonom_memory.md)")

    col_mem1, col_mem2 = st.columns([1, 3])

    with col_mem1:
        if st.button("Hafızayı Yükle"):
            text = load_workspace_memory_file(Path(st.session_state["workspace"]))
            st.session_state["workspace_memory_text"] = text
        if st.button("Hafızayı Kaydet"):
            err = save_workspace_memory_file(
                Path(st.session_state["workspace"]),
                st.session_state["workspace_memory_text"],
            )
            if err is None:
                st.success("Hafıza dosyası kaydedildi.")
            else:
                st.error(f"Hafıza dosyası kaydedilemedi: {err}")

    with col_mem2:
        st.session_state["workspace_memory_text"] = st.text_area(
            "Proje notları / stack / stil kuralları",
            value=st.session_state["workspace_memory_text"],
            height=180,
        )

    st.markdown("---")

    # Manuel Komut Paneli
    st.subheader("🔧 Manuel Komut Çalıştır")

    col_cmd1, col_cmd2 = st.columns([3, 1])
    with col_cmd1:
        st.session_state["manual_command"] = st.text_input(
            "Komut (workspace içinde çalışır)",
            value=st.session_state["manual_command"],
            help="Örn: pytest, python main.py, ruff ., black .",
        )
    with col_cmd2:
        if st.button("Komutu Çalıştır"):
            cmd = st.session_state["manual_command"]
            rc, out, err = run_generic_command(
                workspace=Path(st.session_state["workspace"]),
                command=cmd,
                timeout_seconds=600,
            )
            summary = summarize_generic_command_result(
                "Manuel komut", rc, out, err, cmd
            )
            st.session_state["manual_command_history"].append(summary)

    if st.session_state["manual_command_history"]:
        st.markdown("Son komut çıktıları:")
        for idx, s in enumerate(reversed(st.session_state["manual_command_history"][-5:]), start=1):
            with st.expander(f"Komut Çıktısı #{idx}", expanded=False):
                st.markdown(s)

    st.markdown("---")

    # Chat geçmişi
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input(
        "Görevini yaz (ör: Basit bir FastAPI projesi kur, testler pytest ile çalışsın)."
    )

    if user_input and cfg is not None:
        st.session_state["messages"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        workspace_path = Path(st.session_state["workspace"])
        dry_run_flag = st.session_state["dry_run"]
        run_tests_flag = st.session_state["run_tests"]
        test_cmd = st.session_state["test_command"]
        run_format_flag = st.session_state["run_format"]
        fmt_cmd = st.session_state["format_command"]
        run_lint_flag = st.session_state["run_lint"]
        lint_cmd = st.session_state["lint_command"]
        auto_install_flag = st.session_state["auto_install"]
        pip_cmd_template = st.session_state["pip_install_command"]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown(
                "_Görev işleniyor: plan → kod → (format/lint) → (opsiyonel) test → (gerekirse) auto pip install + düzeltme → review..._"
            )

            try:
                actions, summary = run_agent_round(
                    config=cfg,
                    task=user_input,
                    workspace=workspace_path,
                    mode_override=st.session_state["current_mode"],
                    dry_run=dry_run_flag,
                    run_tests_flag=run_tests_flag,
                    test_command_override=test_cmd,
                    run_format_flag=run_format_flag,
                    format_command_override=fmt_cmd,
                    run_lint_flag=run_lint_flag,
                    lint_command_override=lint_cmd,
                    run_auto_install_flag=auto_install_flag,
                    pip_install_command_override=pip_cmd_template,
                )
                st.session_state["last_actions"] = actions
                placeholder.markdown(summary)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": summary}
                )
            except Exception as e:
                error_text = f"❌ Hata: {e}"
                placeholder.markdown(error_text)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": error_text}
                )

    st.markdown("---")
    st.subheader("Son Dosya Aksiyonları")

    actions = st.session_state.get("last_actions", [])
    if not actions:
        st.info("Henüz dosya aksiyonu yok.")
    else:
        for a in actions:
            path = a.get("path", "?")
            op = a.get("action", "?")
            desc = a.get("description", "")
            if op == "write":
                op_label = "YAZ"
            elif op == "delete":
                op_label = "SİL"
            else:
                op_label = op.upper()
            with st.expander(f"[{op_label}] {path}", expanded=False):
                st.write(f"**Aksiyon:** `{op}`")
                if desc:
                    st.write(f"**Açıklama:** {desc}")
                if op == "write":
                    content = a.get("content", "")
                    st.code(content, language="python")


if __name__ == "__main__":
    main()
