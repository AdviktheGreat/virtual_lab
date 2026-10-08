"""How the web interface looks: a lab notebook by day, a control room by night.

The light theme is paper and ink, with the discussion set in a serif, as the exported documents
are. The dark theme is a control room's: deep slate, cyan for what is live, amber for what waits
on the person. Which one shows follows the system unless the person chooses, and the choice is
kept in the browser.
"""

from typing import Any

THEME_STORAGE_KEY = "virtual-lab-theme"

CSS = """
:root, body {
  --vl-bg: #f6f3ea; --vl-panel: #fffdf7; --vl-panel-2: #f1ede2; --vl-border: #e2dccd;
  --vl-text: #23211d; --vl-muted: #7c766a; --vl-accent: #2f6f8f; --vl-accent-soft: #e3eef3;
  --vl-ok: #3d7a33; --vl-ok-soft: #e3efe0; --vl-bad: #b0413e; --vl-bad-soft: #f6e1df;
  --vl-ask: #a86a12; --vl-ask-soft: #fbefd8; --vl-live: #2f6f8f; --vl-live-soft: #e3eef3;
  --vl-code-bg: #f3efe5; --vl-human: #f5f1e6;
  --vl-body-font: "Charter", "Iowan Old Style", "Georgia", "DejaVu Serif", serif;
  --vl-ui-font: "Inter", "SF Pro Text", "Helvetica Neue", "Segoe UI", system-ui, sans-serif;
  --vl-mono: "JetBrains Mono", "SF Mono", "Menlo", "Consolas", monospace;
}
body.dark {
  --vl-bg: #0b1016; --vl-panel: #111922; --vl-panel-2: #17212c; --vl-border: #233140;
  --vl-text: #dbe4ee; --vl-muted: #7f91a4; --vl-accent: #4cc3d9; --vl-accent-soft: #11303a;
  --vl-ok: #4fd18b; --vl-ok-soft: #10301f; --vl-bad: #ff7b72; --vl-bad-soft: #3a1717;
  --vl-ask: #f5b942; --vl-ask-soft: #3a2a0c; --vl-live: #4cc3d9; --vl-live-soft: #11303a;
  --vl-code-bg: #0d141b; --vl-human: #1b2430;
  --vl-body-font: var(--vl-ui-font);
}

.gradio-container { max-width: 1680px !important; font-family: var(--vl-ui-font); }
footer { display: none !important; }

/* The header */
#vl-header { display: flex; align-items: center; gap: 14px; padding: 6px 2px 2px; }
#vl-header .vl-mark { flex: none; width: 34px; height: 34px; border-radius: 9px; display: grid; place-items: center;
  background: var(--vl-accent); color: var(--vl-panel); font: 700 15px var(--vl-ui-font); letter-spacing: -0.04em; }
#vl-header .vl-name { font: 700 19px var(--vl-ui-font); letter-spacing: -0.02em; color: var(--vl-text); }
#vl-header .vl-tagline { font-size: 12.5px; color: var(--vl-muted); }
#vl-header-row { align-items: center !important; }
#vl-theme-toggle { align-self: center; flex: none; height: 34px; min-width: 120px; max-width: 150px; }

#vl-history-table table, #vl-history-table td, #vl-history-table th { font-family: var(--vl-ui-font) !important; }

/* Panels the run is shown in */
.vl-panel { background: var(--vl-panel); border: 1px solid var(--vl-border); border-radius: 12px; }
#vl-meeting-feed .html-container, #vl-project-feed .html-container { max-height: 74vh; overflow-y: auto !important;
  padding: 4px 14px 18px; }
.vl-feed-empty, .vl-empty { color: var(--vl-muted); font-style: italic; padding: 10px 0; }
.vl-hero { padding: 42px 20px; text-align: center; color: var(--vl-muted); }
.vl-hero h2 { font: 600 20px var(--vl-ui-font); color: var(--vl-text); margin: 0 0 6px; border: none; }

.vl-eyebrow { font: 600 10.5px var(--vl-ui-font); text-transform: uppercase; letter-spacing: 0.12em;
  color: var(--vl-muted); display: flex; align-items: center; gap: 8px; }
.vl-label { font: 700 10.5px var(--vl-ui-font); text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--vl-muted); margin: 14px 0 5px; }
.vl-agenda { border-bottom: 1px solid var(--vl-border); padding: 10px 0 14px; margin-bottom: 6px; }
.vl-agenda-text { font-family: var(--vl-body-font); font-size: 15.5px; line-height: 1.5; color: var(--vl-text); }
.vl-agenda-text p { margin: 6px 0; }
.vl-questions { margin: 4px 0 8px 18px; font-family: var(--vl-body-font); color: var(--vl-text); }
.vl-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.vl-chip { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  border: 1px solid var(--vl-border); background: var(--vl-panel-2); font: 500 12px var(--vl-ui-font);
  color: var(--vl-text); }
.vl-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: none; }
.vl-dot.vl-you { background: var(--vl-text); }
.vl-model { color: var(--vl-muted); font-weight: 400; font-size: 11px; margin-left: 4px; }

.vl-round { display: flex; align-items: center; gap: 10px; margin: 22px 0 6px; font: 700 10.5px var(--vl-ui-font);
  text-transform: uppercase; letter-spacing: 0.14em; color: var(--vl-muted); }
.vl-round::after { content: ""; flex: 1; border-top: 1px solid var(--vl-border); }

.vl-turn { border-left: 3px solid var(--colour, var(--vl-border)); padding: 4px 0 4px 14px; margin: 12px 0;
  animation: vl-in 0.25s ease-out; }
.vl-speaker { display: flex; align-items: center; gap: 7px; font: 650 13px var(--vl-ui-font); color: var(--colour); }
body.dark .vl-speaker { color: color-mix(in srgb, var(--colour) 45%, white); }
body.dark .vl-turn { border-left-color: color-mix(in srgb, var(--colour) 60%, white); }
.vl-what { color: var(--vl-muted); font-weight: 400; }
.vl-body { font-family: var(--vl-body-font); font-size: 15px; line-height: 1.6; color: var(--vl-text);
  overflow-wrap: anywhere; }
.vl-body p { margin: 6px 0; }
.vl-body h1, .vl-body h2, .vl-body h3, .vl-body h4 { font: 650 14.5px var(--vl-ui-font); margin: 14px 0 4px;
  border: none; padding: 0; color: var(--vl-text); }
.vl-body table { border-collapse: collapse; font-size: 13px; margin: 8px 0; }
.vl-body th, .vl-body td { border-bottom: 1px solid var(--vl-border); padding: 4px 8px; text-align: left; }
.vl-body code, .vl-call code { font-family: var(--vl-mono); font-size: 12.5px; background: var(--vl-code-bg);
  border-radius: 4px; padding: 0 4px; }
.vl-body pre code { background: none; padding: 0; }
.vl-body img { max-width: 100%; }
.vl-body blockquote { border-left: 3px solid var(--vl-border); margin: 6px 0; padding: 0 12px; color: var(--vl-muted); }
pre.vl-code, pre.vl-output, .vl-body pre { font-family: var(--vl-mono); font-size: 12.5px; line-height: 1.5;
  background: var(--vl-code-bg); border: 1px solid var(--vl-border); border-radius: 8px; padding: 10px 12px;
  white-space: pre-wrap; overflow-wrap: anywhere; color: var(--vl-text); max-height: 420px; overflow: auto; }
.vl-turn.vl-human { --colour: var(--vl-text); background: var(--vl-human); border-radius: 0 10px 10px 0;
  padding: 8px 14px; }
.vl-turn.vl-human .vl-speaker { color: var(--vl-text); }
.vl-turn.vl-tools .vl-calls { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 6px;
  font: 13px var(--vl-ui-font); color: var(--vl-muted); }
.vl-call { display: inline-flex; gap: 6px; align-items: baseline; border: 1px solid var(--vl-border);
  border-radius: 8px; padding: 2px 8px; background: var(--vl-panel-2); max-width: 100%; }
.vl-args { color: var(--vl-muted); font: 11.5px var(--vl-mono); overflow-wrap: anywhere; }

.vl-fold { margin: 8px 0; border: 1px solid var(--vl-border); border-radius: 10px; background: var(--vl-panel); }
.vl-fold > summary { cursor: pointer; padding: 7px 12px; font: 500 13px var(--vl-ui-font); color: var(--vl-muted);
  list-style: none; display: flex; align-items: center; gap: 8px; }
.vl-fold > summary::-webkit-details-marker { display: none; }
.vl-fold > summary::before { content: "›"; font-size: 15px; transition: transform 0.15s; display: inline-block; }
.vl-fold[open] > summary::before { transform: rotate(90deg); }
.vl-fold-body { padding: 0 12px 10px; }
.vl-fold.vl-prompt { border: none; background: transparent; margin: 2px 0; }
.vl-fold.vl-prompt > summary { font-size: 11.5px; padding: 2px 4px; opacity: 0.7; }
.vl-fold.vl-meeting { background: var(--vl-panel-2); }
.vl-fold.vl-meeting > summary { color: var(--vl-text); font-weight: 600; }
.vl-fold.vl-meeting > .vl-fold-body { background: var(--vl-panel); border-radius: 0 0 10px 10px; padding-top: 4px; }

.vl-summary { margin: 22px 0 6px; border: 1px solid var(--vl-accent); border-radius: 12px; padding: 12px 16px;
  background: var(--vl-accent-soft); }
.vl-failed { margin: 16px 0; border: 1px solid var(--vl-bad); background: var(--vl-bad-soft); color: var(--vl-text);
  border-radius: 10px; padding: 10px 14px; font-size: 14px; }
.vl-ended { margin: 16px 0; color: var(--vl-muted); font: 500 13px var(--vl-ui-font); text-align: center; }
.vl-marker { margin: 10px 0; padding: 7px 12px; border-radius: 8px; background: var(--vl-panel-2);
  font: 13.5px var(--vl-ui-font); color: var(--vl-text); }
.vl-marker.vl-decided { border-left: 3px solid var(--vl-accent); }
.vl-marker.vl-quiet { background: transparent; color: var(--vl-muted); padding: 2px 12px; }
.vl-cell { margin: 8px 0; font: 13px var(--vl-ui-font); color: var(--vl-muted); }
.vl-figure { margin: 10px 0; }
.vl-figure img { max-width: 100%; max-height: 460px; border-radius: 8px; border: 1px solid var(--vl-border);
  background: white; }
.vl-figure figcaption { font-size: 12px; color: var(--vl-muted); margin-top: 3px; }
.vl-error-text { color: var(--vl-bad); font: 13px var(--vl-ui-font); margin: 6px 0; overflow-wrap: anywhere; }

.vl-badge { display: inline-flex; align-items: center; gap: 5px; border-radius: 999px; padding: 1px 9px;
  font: 650 11px var(--vl-ui-font); background: var(--vl-panel-2); color: var(--vl-muted); white-space: nowrap; }
.vl-badge.ok { background: var(--vl-ok-soft); color: var(--vl-ok); }
.vl-badge.bad { background: var(--vl-bad-soft); color: var(--vl-bad); }
.vl-badge.ask { background: var(--vl-ask-soft); color: var(--vl-ask); }
.vl-badge.live { background: var(--vl-live-soft); color: var(--vl-live); }
.vl-badge.live::before, .vl-badge.ask::before { content: ""; width: 6px; height: 6px; border-radius: 50%;
  background: currentColor; animation: vl-pulse 1.4s ease-in-out infinite; }

/* The rail beside a run */
.vl-state { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.vl-elapsed { font: 12px var(--vl-mono); color: var(--vl-muted); }
.vl-doing { font: 600 15px var(--vl-ui-font); color: var(--vl-text); margin: 10px 0 4px; }
.vl-meter { height: 6px; border-radius: 999px; background: var(--vl-panel-2); overflow: hidden;
  border: 1px solid var(--vl-border); }
.vl-meter span { display: block; height: 100%; background: var(--vl-ok); transition: width 0.4s; }
.vl-meter.ask span { background: var(--vl-ask); }
.vl-meter.bad span { background: var(--vl-bad); }
.vl-notes { margin: 4px 0 0 16px; font-size: 13px; color: var(--vl-text); }

.vl-ask { border: 1px solid var(--vl-ask); background: var(--vl-ask-soft); border-radius: 12px; padding: 12px 14px; }
.vl-ask h3 { font: 600 15px var(--vl-ui-font); margin: 6px 0; color: var(--vl-text); border: none; }
.vl-ask .vl-body { font-size: 14px; }

/* The project's board */
.vl-goal { font-family: var(--vl-body-font); font-size: 16px; color: var(--vl-text); }
.vl-plan { list-style: none; padding: 0; margin: 8px 0 0; font-size: 14px; color: var(--vl-text); }
.vl-plan li { padding: 3px 0 3px 24px; position: relative; }
.vl-plan li::before { position: absolute; left: 0; width: 18px; text-align: center; }
.vl-plan li.done::before { content: "✓"; color: var(--vl-ok); font-weight: 700; }
.vl-plan li.progress::before { content: "◐"; color: var(--vl-ask); }
.vl-plan li.todo::before { content: "○"; color: var(--vl-muted); }
.vl-plan li.dropped { color: var(--vl-muted); text-decoration: line-through; }
.vl-plan li.dropped::before { content: "×"; color: var(--vl-bad); }
.vl-rounds { list-style: none; padding: 0; margin: 6px 0; font-size: 13.5px; color: var(--vl-text); }
.vl-rounds li { padding: 3px 0; }
.vl-rounds li.vl-outcome { color: var(--vl-muted); padding-left: 30px; font-size: 12.5px; }
.vl-round-number { display: inline-grid; place-items: center; width: 20px; height: 20px; border-radius: 50%;
  margin-right: 8px; background: var(--vl-accent-soft); color: var(--vl-accent); font: 700 11px var(--vl-ui-font); }
.vl-report { border: 1px solid var(--vl-accent); background: var(--vl-accent-soft); border-radius: 12px;
  padding: 12px 14px; margin: 12px 0; color: var(--vl-text); }

/* Settings */
.vl-keys { display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; font: 13.5px var(--vl-ui-font);
  color: var(--vl-text); }
.vl-keys code { font-family: var(--vl-mono); font-size: 12px; }

.vl-pending .vl-thinking { display: inline-flex; gap: 4px; padding: 8px 0 2px; }
.vl-thinking span { width: 6px; height: 6px; border-radius: 50%; background: var(--colour, var(--vl-muted));
  animation: vl-bounce 1.2s infinite ease-in-out; opacity: 0.5; }
.vl-thinking span:nth-child(2) { animation-delay: 0.15s; }
.vl-thinking span:nth-child(3) { animation-delay: 0.3s; }
.vl-caret { display: inline-block; width: 7px; height: 1.05em; margin-left: 2px; vertical-align: text-bottom;
  background: var(--vl-accent); animation: vl-blink 1s steps(1) infinite; }
.vl-writing .vl-body > p:last-of-type { display: inline; }
.vl-spinner { width: 12px; height: 12px; border: 2px solid var(--vl-border); border-top-color: var(--vl-accent);
  border-radius: 50%; display: inline-block; animation: vl-spin 0.8s linear infinite; }
@keyframes vl-in { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: none; } }
@keyframes vl-pulse { 0%, 100% { opacity: 0.35; } 50% { opacity: 1; } }
@keyframes vl-bounce { 0%, 80%, 100% { transform: translateY(0); } 40% { transform: translateY(-4px); opacity: 1; } }
@keyframes vl-blink { 50% { opacity: 0; } }
@keyframes vl-spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""

# Runs once the page has loaded: the theme as the person chose it, and the discussion kept as
# they left it, scrolled to the end while they are there and its folds as they opened them,
# while it is shown again with more in it
JS = """
() => {
  const KEY = "%(key)s";
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const apply = (mode) => {
    const dark = mode === "dark" || (mode !== "light" && media.matches);
    document.body.classList.toggle("dark", dark);
    document.documentElement.dataset.vlTheme = mode;
  };
  window.vlTheme = () => localStorage.getItem(KEY) || "system";
  window.vlSetTheme = (mode) => {
    mode = ["light", "dark", "system"].includes(mode) ? mode : "system";
    localStorage.setItem(KEY, mode);
    apply(mode);
    return mode;
  };
  window.vlToggleTheme = () => window.vlSetTheme(document.body.classList.contains("dark") ? "light" : "dark");
  apply(window.vlTheme());
  media.addEventListener("change", () => setTimeout(() => apply(window.vlTheme()), 0));

  const folds = {};
  document.addEventListener("click", (event) => {
    const summary = event.target.closest && event.target.closest("summary");
    const fold = summary && summary.parentElement;
    if (fold && fold.dataset && fold.dataset.key) folds[fold.dataset.key] = !fold.open;
  }, true);
  const restore = (root) => {
    root.querySelectorAll("details[data-key]").forEach((fold) => {
      const open = folds[fold.dataset.key];
      if (open !== undefined && fold.open !== open) fold.open = open;
    });
  };
  const follow = (id) => {
    const feed = document.querySelector("#" + id + " .html-container");
    if (!feed || feed.dataset.vlFollowing) return;
    feed.dataset.vlFollowing = "yes";
    let atEnd = true;
    feed.addEventListener("scroll", () => {
      atEnd = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 120;
    });
    new MutationObserver(() => {
      restore(feed);
      if (atEnd) feed.scrollTop = feed.scrollHeight;
    }).observe(feed, { childList: true, subtree: true });
  };
  setInterval(() => ["vl-meeting-feed", "vl-project-feed"].forEach(follow), 500);
  return window.vlTheme();
}
""" % {"key": THEME_STORAGE_KEY}


def gradio_theme() -> Any:
    """Gradio's own components in the same colours, by day and by night."""
    import gradio as gr

    sans = ["Inter", "SF Pro Text", "Helvetica Neue", "Segoe UI", "system-ui", "sans-serif"]
    mono = ["JetBrains Mono", "SF Mono", "Menlo", "Consolas", "monospace"]
    teal = gr.themes.Color(
        c50="#eef6f9",
        c100="#d8eaf1",
        c200="#b3d4e1",
        c300="#86b9cd",
        c400="#5a9bb5",
        c500="#2f6f8f",
        c600="#285f7b",
        c700="#214f66",
        c800="#1a3f52",
        c900="#132f3d",
        c950="#0c1f29",
    )

    return gr.themes.Base(primary_hue=teal, secondary_hue="stone", neutral_hue="stone", font=sans, font_mono=mono).set(
        body_background_fill="#f6f3ea",
        body_background_fill_dark="#0b1016",
        background_fill_primary="#fffdf7",
        background_fill_primary_dark="#111922",
        background_fill_secondary="#f1ede2",
        background_fill_secondary_dark="#17212c",
        block_background_fill="#fffdf7",
        block_background_fill_dark="#111922",
        block_border_color="#e2dccd",
        block_border_color_dark="#233140",
        border_color_primary="#e2dccd",
        border_color_primary_dark="#233140",
        block_label_text_color="#7c766a",
        block_label_text_color_dark="#7f91a4",
        block_title_text_color="#23211d",
        block_title_text_color_dark="#dbe4ee",
        body_text_color="#23211d",
        body_text_color_dark="#dbe4ee",
        body_text_color_subdued="#7c766a",
        body_text_color_subdued_dark="#7f91a4",
        input_background_fill="#f8f5ec",
        input_background_fill_dark="#0d141b",
        input_border_color="#d6cfbb",
        input_border_color_dark="#233140",
        input_border_width="1px",
        button_primary_background_fill="#2f6f8f",
        button_primary_background_fill_hover="#285f7b",
        button_primary_background_fill_dark="#1f8aa0",
        button_primary_background_fill_hover_dark="#26a2bb",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#06121a",
        button_secondary_background_fill="#f1ede2",
        button_secondary_background_fill_dark="#17212c",
        button_secondary_text_color="#23211d",
        button_secondary_text_color_dark="#dbe4ee",
        button_cancel_background_fill="#f6e1df",
        button_cancel_background_fill_dark="#3a1717",
        button_cancel_text_color="#8e2d27",
        button_cancel_text_color_dark="#ff9b94",
        block_radius="12px",
        button_large_radius="10px",
        button_small_radius="8px",
        input_radius="8px",
        checkbox_label_background_fill_selected="#d8eaf1",
        checkbox_label_background_fill_selected_dark="#11303a",
    )
