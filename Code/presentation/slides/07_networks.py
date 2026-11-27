"""LSTM and GRU: equations, our architecture, a parameter calculator and a one-cell playground."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import loaders as L
import theme

KEY = "07_networks"
theme.header(KEY)

cfg = L.config()
m = cfg["model"]
n_all = L.levels().shape[1]

eq, arch = st.columns([1.1, 1], gap="large")
with eq:
    t_lstm, t_gru = st.tabs(["LSTM cell", "GRU cell"])
    with t_lstm:
        st.latex(r"""\begin{aligned}
        f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) && \text{forget gate}\\
        i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) && \text{input gate}\\
        o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) && \text{output gate}\\
        \tilde c_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) && \text{candidate}\\
        c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t && \text{cell state}\\
        h_t &= o_t \odot \tanh(c_t) && \text{hidden state}
        \end{aligned}""")
        st.caption("Hochreiter and Schmidhuber (1997). The additive update of c_t lets gradients flow "
                   "across many time steps instead of vanishing.")
    with t_gru:
        st.latex(r"""\begin{aligned}
        z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) && \text{update gate}\\
        r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) && \text{reset gate}\\
        \tilde h_t &= \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) && \text{candidate}\\
        h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t && \text{hidden state}
        \end{aligned}""")
        st.caption("Cho et al. (2014). Two gates instead of three and no separate cell state: about 25% fewer "
                   "weights than an LSTM of the same width.")

with arch:
    st.subheader("Our forecaster")
    st.graphviz_chart(f"""
    digraph {{
      rankdir=TB; nodesep=0.3; ranksep=0.2;
      node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="{theme.RULE}", fontname="Helvetica",
            fontsize=11, fontcolor="{theme.INK}", margin="0.2,0.07"];
      edge [color="{theme.MUTED}", arrowsize=0.6];
      x [label="Input window: {cfg['framing']['lookback']} days x n features (scaled daily changes)"];
      r1 [label="{m['type'].upper()} layer 1, {m['hidden_size']} units", fillcolor="#EAF2FC"];
      r2 [label="{m['type'].upper()} layer 2, {m['hidden_size']} units, dropout {m['dropout']} between layers", fillcolor="#EAF2FC"];
      hT [label="Last hidden state h_T"];
      head [label="Dropout {m['dropout']}, then Linear({m['hidden_size']} -> 1)"];
      y [label="Predicted h-day log return"];
      p [label="Price forecast  P_t x exp(prediction)", fillcolor="{theme.CRUDE_SOFT}"];
      x -> r1 -> r2 -> hT -> head -> y -> p;
    }}""", width="stretch")
    st.caption("oilfc/models.py: RNNForecaster. One model per horizon (direct strategy); GRU swaps the cell type.")

st.subheader("How many weights?")
c1, c2 = st.columns([1, 1.4], gap="large")
with c1:
    cell = st.segmented_control("Cell", ["lstm", "gru"], default=m["type"], format_func=str.upper, required=True)
    hidden = st.select_slider("Hidden units per layer", [16, 32, 64, 128, 256], value=m["hidden_size"])
    layers = st.select_slider("Layers", [1, 2, 3, 4], value=m["num_layers"])
    n_in = st.segmented_control("Input set", {f"Price only (1)": 1, f"Price + market drivers ({n_all})": n_all,
                                              f"Price + satellite supply (4)": 4}.keys(),
                                default=f"Price + market drivers ({n_all})", required=True)
    n_in = int(n_in.split("(")[1].rstrip(")"))


def rnn_params(cell, n_in, hidden, layers):
    """PyTorch counts two bias vectors per gate (b_ih and b_hh)."""
    g = 4 if cell == "lstm" else 3
    per = []
    for k in range(layers):
        inp = n_in if k == 0 else hidden
        per.append(g * hidden * (inp + hidden) + 2 * g * hidden)
    return per, hidden + 1


with c2:
    per, head = rnn_params(cell, n_in, hidden, layers)
    total = sum(per) + head
    fig = go.Figure()
    for k, p in enumerate(per):
        fig.add_trace(go.Bar(y=["weights"], x=[p], orientation="h", name=f"Layer {k + 1}",
                             marker_color=[theme.BLUE, "#6da7ec", "#9ec5f4", "#cde2fb"][k],
                             marker_line=dict(color="#fff", width=2),
                             hovertemplate=f"Layer {k + 1}: %{{x:,}}<extra></extra>"))
    fig.add_trace(go.Bar(y=["weights"], x=[head], orientation="h", name="Linear head", marker_color=theme.CRUDE,
                         hovertemplate="Head: %{x:,}<extra></extra>"))
    fig.update_layout(barmode="stack", height=170, legend_traceorder="normal", yaxis_visible=False, xaxis_title="trainable parameters",
                      margin=dict(t=40, b=10))
    theme.show(fig)
    windows = int((L.levels().index <= L.split_dates()[0]).sum()) - cfg["framing"]["lookback"]
    theme.figures([(f"{total:,}", "trainable parameters"),
                   (f"{windows:,}", "training windows (about)"),
                   (f"{total / windows:.0f}", "weights per training example")])
    runs = L.main_runs()
    if not runs.empty and cell == m["type"] and hidden == m["hidden_size"] and layers == m["num_layers"] \
            and n_in == n_all:
        st.caption(f"Matches the {runs.n_params.iloc[0]:,} parameters reported by the trained runs. "
                   "Far more weights than examples: dropout and early stopping keep this from memorising.")

with st.expander("Play with one LSTM cell: what does the forget gate do?"):
    st.write("One unit, one time step. Move the gates and see how much of the old memory survives.")
    a, b, c = st.columns(3)
    c_prev = a.slider("Old memory c(t-1)", -2.0, 2.0, 1.5, 0.1)
    cand = a.slider("New candidate c~(t)", -1.0, 1.0, -0.6, 0.1)
    f = b.slider("Forget gate f", 0.0, 1.0, 0.9, 0.05)
    i = b.slider("Input gate i", 0.0, 1.0, 0.2, 0.05)
    o = c.slider("Output gate o", 0.0, 1.0, 0.8, 0.05)
    c_t = f * c_prev + i * cand
    h_t = o * np.tanh(c_t)
    fig = go.Figure(go.Bar(x=["kept: f x c(t-1)", "written: i x c~(t)", "new memory c(t)", "output h(t)"],
                           y=[f * c_prev, i * cand, c_t, h_t],
                           marker_color=[theme.BLUE, theme.ORANGE, theme.INK, theme.CRUDE],
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>", text=[f"{v:+.2f}" for v in
                                                                                  (f * c_prev, i * cand, c_t, h_t)],
                           textposition="outside"))
    fig.add_hline(y=0, line=dict(color=theme.MUTED, width=1))
    fig.update_layout(yaxis_range=[-3, 3], showlegend=False)
    c.write("")
    theme.show(fig, height=280)
    st.caption("With f near 1 and i near 0 the cell carries its memory forward untouched, which is how an "
               "LSTM can hold information across a 30-day window.")

theme.footer(KEY)
