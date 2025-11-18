#!/usr/bin/env python
"""
Streamlit tabanlı web arayüzü (plan → apply + test fix, v4).

Özellikler:
- Chat arayüzü ile görev verme
- Cloud / local mod seçimi
- Workspace klasörü seçimi / oluşturma
- Dry-run togglesi (sadece plan/aksiyon üret)
- Testleri çalıştırma (pytest vb.) + oto-fix döngüsü
- Son dosya aksiyonlarını detaylı görme
- Workspace dosya gezgini (dosya seç → içeriğini gör)
- Planner + Coder + Test özetini tek mesajda görme
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from otonom_coder_agent import (
    AppConfig,
    load_config,
    run_agent_round,
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
    if "selected_file" not in st.session_state:
        st.session_state["selected_file"] = ""


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


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config("Otonom Coder", layout="wide")
    init_session_state()

    st.title("🧠 Otonom Coder (Qwen3-Coder + Ollama)")

    # Sidebar: config, mod, workspace, dry-run, test
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

        st.markdown("---")
        st.caption("Görevleri aşağıdaki sohbet kutusundan yaz.")

    cfg: Optional[AppConfig] = st.session_state["config"]

    # Üst bilgi ve workspace explorer
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
            st.write(
                f"**Max test fix turu:** `{cfg.general.max_test_fix_rounds}`"
            )

        st.markdown(
            "_İpucu: max_rounds ≥ 2 ise her görevde önce plan çıkar, sonra dosya aksiyonları üretilir; test açık ise hata varsa kendisi düzeltmeye çalışır._"
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
                # Dil tahmini yok; çoğu kod olacağı için python bırakıyorum
                st.code(content, language="python")

    st.markdown("---")

    # Chat geçmişi
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Görevini yaz (ör: Basit bir FastAPI projesi kur, testler pytest ile çalışsın).")

    if user_input and cfg is not None:
        # Kullanıcı mesajını ekle
        st.session_state["messages"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Agent çalıştır
        workspace_path = Path(st.session_state["workspace"])
        dry_run_flag = st.session_state["dry_run"]
        run_tests_flag = st.session_state["run_tests"]
        test_cmd = st.session_state["test_command"]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown(
                "_Görev işleniyor: plan → kod → (opsiyonel) test → (gerekirse) düzeltme..._"
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

    # Son aksiyonları tabloda göster
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
