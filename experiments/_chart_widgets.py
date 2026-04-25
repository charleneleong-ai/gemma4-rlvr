"""Reusable HTML/CSS/JS widgets injected into static Plotly charts via
`fig.write_html(post_script=...)`.

Standalone Plotly HTML files can't use Dash widgets (`dash_daq.BooleanSwitch`,
`dbc.Switch`, etc.) because those require a running Dash app server. Plotly's
native `updatemenus` only renders rectangular buttons and dropdowns — no
toggle/switch primitive. So we generate small inline widgets here.

Each function returns a string suitable for the `post_script` parameter of
`plotly.graph_objects.Figure.write_html`. Plotly substitutes `{plot_id}` at
write time with the chart's div id.
"""
from __future__ import annotations

import json
from typing import Iterable

# CSS shared by every iOS-style switch on the page. Scoped under
# `.cw-switch` so callers won't collide with their own styles.
_SWITCH_CSS = """
.cw-switch-wrap {
    position: absolute; z-index: 1000;
    display: flex; align-items: center; gap: 8px;
    font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #444;
    background: rgba(255, 255, 255, 0.92);
    padding: 6px 12px;
    border: 1px solid #ddd;
    border-radius: 14px;
    user-select: none;
}
.cw-switch { position: relative; display: inline-block; width: 34px; height: 18px; }
.cw-switch input { opacity: 0; width: 0; height: 0; }
.cw-switch .slider {
    position: absolute; cursor: pointer; inset: 0;
    transition: 0.2s; border-radius: 18px;
}
.cw-switch .slider::before {
    position: absolute; content: ""; height: 14px; width: 14px;
    bottom: 2px; transition: 0.2s; border-radius: 50%;
    background: white;
}
.cw-switch input:checked + .slider::before { left: 18px; }
.cw-switch input:not(:checked) + .slider::before { left: 2px; }
"""

_POSITIONS = {
    # Sit just below the chart title (margin t≈130, title font ≈22px), well
    # above any subplot subtitle so we don't block per-task descriptions.
    "top-left": "top: 42px; left: 24px;",
    "top-right": "top: 42px; right: 24px;",
    "bottom-left": "bottom: 24px; left: 24px;",
    "bottom-right": "bottom: 24px; right: 24px;",
}


def plotly_label_toggle(
    *,
    label_indices: Iterable[int],
    n_traces: int,
    label: str = "labels",
    position: str = "top-left",
    on_color: str = "#2ecc71",
    off_color: str = "#ccc",
    default_on: bool = True,
) -> str:
    """Post-script for a switch that toggles per-row Plotly annotations
    AND flips marker `hoverinfo` between `skip` (labels visible, dot
    hover dormant) and `text` (labels hidden, dot hover active).

    Wire into the figure with::

        fig.write_html(path, post_script=plotly_label_toggle(
            label_indices=label_indices,
            n_traces=len(fig.data),
        ))
    """
    if position not in _POSITIONS:
        raise ValueError(f"position must be one of {list(_POSITIONS)}")
    pos_css = _POSITIONS[position]
    indices_json = json.dumps(list(label_indices))
    n_traces_json = json.dumps(n_traces)
    checked = "checked" if default_on else ""
    css = _SWITCH_CSS + f"""
        .cw-switch input:checked + .slider {{ background: {on_color}; }}
        .cw-switch input:not(:checked) + .slider {{ background: {off_color}; }}
    """
    return f"""
    (function () {{
        const gd = document.getElementById('{{plot_id}}');
        if (!gd) return;
        const style = document.createElement('style');
        style.textContent = `{css}`;
        document.head.appendChild(style);

        const wrap = document.createElement('label');
        wrap.className = 'cw-switch-wrap';
        wrap.style.cssText = `{pos_css}`;
        wrap.innerHTML = `
            <span>{label}</span>
            <span class="cw-switch">
                <input type="checkbox" {checked}>
                <span class="slider"></span>
            </span>
        `;
        gd.parentNode.style.position = 'relative';
        gd.parentNode.appendChild(wrap);

        const LABEL_INDICES = {indices_json};
        const N_TRACES = {n_traces_json};
        wrap.querySelector('input').addEventListener('change', (e) => {{
            const on = e.target.checked;
            const layoutUpdate = {{}};
            for (const i of LABEL_INDICES) {{
                layoutUpdate[`annotations[${{i}}].visible`] = on;
            }}
            Plotly.relayout(gd, layoutUpdate);
            Plotly.restyle(gd,
                {{hoverinfo: on ? 'skip' : 'text'}},
                [...Array(N_TRACES).keys()],
            );
        }});
    }})();
    """
