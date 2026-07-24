//! Behaviour: arrow-navigating the orchestrator dock onto a **slow / high-
//! latency** SSH workspace keeps the whole editor responsive while the session
//! connects and materializes in the background.
//!
//! This is the companion to `orchestrator_dock_ssh_navigation_nonblocking.rs`.
//! That test uses a host that never establishes the channel (`fake-ssh-hang`),
//! so `is_connected()` stays false and every remote request fails fast — the
//! switch is trivially non-blocking. The bug this test pins is the *opposite*
//! shape: a host that **does** connect but is bandwidth-throttled
//! (`fake-ssh-slow`, cf. `ProxyCommand … | pv -qL 20k`). There the channel
//! comes up and the agent answers, so promoting the dived-into session
//! re-reads its persisted buffers over the slow link. Done on the editor loop,
//! those reads froze the whole UI for seconds — the user could not arrow to
//! another window until the reads returned.
//!
//! The fix prewarms the session's files on the connect worker, off the editor
//! loop, so materialization serves them from cache and the dock stays live.
//!
//! Reproducer shape: the slow shim holds the file `read` open indefinitely (it
//! dribbles keepalive chunks so the request never times out). Without the fix
//! the promote-time read runs on the editor loop and never returns, so driving
//! frames below hangs and nextest's external per-test cap fails the test. With
//! the fix the read runs on the connect worker; the loop keeps turning and the
//! command palette still opens on demand.
//!
//! Single test in this binary: the fake-ssh PATH shim and `isolated_dir_context`'s
//! process-global `XDG_DATA_HOME` / `FAKE_SSH_SLOW_*` env must not leak.
#![cfg(all(target_os = "linux", feature = "plugins"))]

mod common;

use common::dormant_ssh::{
    canonical_mkdir, ensure_slow_fake_ssh_on_path, isolated_dir_context, persist_previous_session,
};
use common::harness::{copy_plugin, copy_plugin_lib, EditorTestHarness, HarnessOptions};
use crossterm::event::{KeyCode, KeyModifiers};

#[test]
fn arrow_nav_onto_slow_remote_keeps_editor_responsive() {
    common::tracing::init_tracing_from_env();
    ensure_slow_fake_ssh_on_path();
    fresh::i18n::set_locale("en");

    let base = tempfile::tempdir().unwrap();
    let dir_context = isolated_dir_context(base.path());
    let project = canonical_mkdir(base.path(), "project");
    let remote_root = canonical_mkdir(base.path(), "remote-root");

    // Throttle the shim: hold every `read` open until this gate file is removed,
    // modelling a transfer that makes only trivial progress (the shim keeps the
    // request alive so it never times out). The gate lives under the per-test
    // temp tree, so it's cleaned up with everything else on teardown — which
    // also releases the held read on the connect worker.
    let gate = base.path().join("read.gate");
    std::fs::write(&gate, "hold").unwrap();
    std::env::set_var("FAKE_SSH_SLOW_METHODS", "read");
    std::env::set_var("FAKE_SSH_SLOW_BLOCK_FILE", &gate);

    let plugins_dir = project.join("plugins");
    std::fs::create_dir_all(&plugins_dir).unwrap();
    copy_plugin_lib(&plugins_dir);
    copy_plugin(&plugins_dir, "orchestrator");

    // Leaves behind a local project workspace + a dormant SSH session
    // (`ssh-dead`) with a real persisted buffer (`remote_notes.txt`) that
    // promoting the session will reopen over the (slow) link.
    persist_previous_session(&dir_context, &project, &remote_root, true);

    let mut cfg = fresh::config::Config::default();
    cfg.editor.animations = false;
    cfg.editor.cursor_jump_animation = false;
    let mut h = EditorTestHarness::create(
        140,
        40,
        HarnessOptions::new()
            .with_config(cfg)
            .with_working_dir(project.clone())
            .with_shared_dir_context(dir_context.clone()),
    )
    .unwrap();
    h.wait_until(|h| {
        let reg = h.editor().command_registry().read().unwrap();
        reg.get_all()
            .iter()
            .any(|c| c.get_localized_name() == "Orchestrator: Toggle Dock")
    })
    .unwrap();
    h.open_file(&project.join("local_marker.txt")).unwrap();
    h.wait_for_screen_contains("local_marker.txt").unwrap();

    // Open the dock.
    h.send_key(KeyCode::Char('p'), KeyModifiers::CONTROL)
        .unwrap();
    h.wait_for_prompt().unwrap();
    h.type_text("Toggle Dock").unwrap();
    h.wait_until(|h| h.screen_to_string().contains("Toggle Dock"))
        .unwrap();
    h.send_key(KeyCode::Enter, KeyModifiers::NONE).unwrap();
    h.wait_until(|h| {
        let scr = h.screen_to_string();
        scr.contains("ssh-dead") && scr.contains("⇅")
    })
    .unwrap();

    // Arrow onto the SSH row: the switch commits into the session's placeholder
    // "Connecting…" page and the connect + prewarm run in the background.
    h.send_key(KeyCode::Down, KeyModifiers::NONE).unwrap();
    h.wait_until(|h| h.editor().active_window().root == remote_root)
        .unwrap();
    h.wait_until(|h| {
        let scr = h.screen_to_string();
        scr.contains("The workspace") || scr.contains("Connecting")
    })
    .unwrap();

    // Drive the event loop while the background connect resolves. WITHOUT the
    // fix the resolving attach re-reads `remote_notes.txt` on THIS loop; the
    // throttled read never returns, so the editor freezes here and the test is
    // killed by nextest's per-test cap. WITH the fix that read happened on the
    // connect worker, so the loop keeps turning past the point the freeze used
    // to occur.
    for _ in 0..30 {
        h.tick_and_render().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    // Still fully responsive to input: the command palette opens on demand and
    // echoes a query, all while the remote is still connecting in the
    // background (the held read means the session never leaves "Connecting…").
    h.send_key(KeyCode::Char('p'), KeyModifiers::CONTROL)
        .unwrap();
    h.wait_for_prompt().unwrap();
    h.type_text("Toggle Dock").unwrap();
    h.wait_until(|h| h.screen_to_string().contains("Toggle Dock"))
        .unwrap();
}
