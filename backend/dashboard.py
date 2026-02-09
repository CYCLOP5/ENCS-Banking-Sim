import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import simulation_engine as sim
from pathlib import Path
from strategic_model import run_game_simulation
from climate_risk import assign_climate_exposure, run_transition_shock
try:
    import torch
    from ml_pipeline import load_trained_model, predict_risk
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
GNN_MODEL_PATH = Path(__file__).parent / "gnn_model.pth"
st.set_page_config(
    page_title="ENCS Systemic Risk Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, 
        border: 1px solid 
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.5rem; font-weight: bold; }
    .metric-label { font-size: 0.9rem; color: #888; }
    .status-safe { color: #00ff88; }
    .status-distressed { color: #ffaa00; }
    .status-default { color: #ff4444; }
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_network_data():
    """Load and prepare network data (cached - runs once)."""
    W_sparse, df = sim.load_and_align_network()
    W_dense = sim.rescale_matrix_to_dollars(W_sparse, df)
    return W_dense, df
@st.cache_resource
def compute_initial_state(_W_dense, _df):
    """Compute initial state (cached)."""
    return sim.compute_state_variables(_W_dense, _df)
def generate_network_layout(df, n_nodes=150):
    """Generate 3D layout for network visualization using spring layout."""
    top_df = df.head(n_nodes).copy()
    G = nx.Graph()
    for i, row in top_df.iterrows():
        G.add_node(i, name=row['bank_name'][:30], 
                   region=row['region'],
                   assets=row['total_assets'])
    for i in range(min(30, len(top_df))):  
        for j in range(30, min(100, len(top_df))):
            if np.random.random() < 0.3:
                G.add_edge(i, j)
    pos_2d = nx.spring_layout(G, k=2.0, iterations=50, seed=42)
    coords = {}
    for node, (x, y) in pos_2d.items():
        region = top_df.iloc[node]['region'] if node < len(top_df) else 'US'
        z = 1.0 if region == 'EU' else 0.0
        z += np.random.uniform(-0.2, 0.2)
        coords[node] = (x, y, z)
    return coords, top_df
def create_3d_network(df, status_array=None, layout_df=None, coords=None):
    """Create 3D Plotly network visualization."""
    if coords is None or layout_df is None:
        coords, layout_df = generate_network_layout(df)
    x_nodes = []
    y_nodes = []
    z_nodes = []
    colors = []
    sizes = []
    hover_texts = []
    for idx, row in layout_df.iterrows():
        if idx in coords:
            x, y, z = coords[idx]
            x_nodes.append(x)
            y_nodes.append(y)
            z_nodes.append(z)
            is_ccp = (row.get('region') == 'Global')
            if status_array is not None and idx < len(status_array):
                status = status_array[idx]
                if is_ccp:
                    colors.append('#FFD700')  
                elif status == 'Default':
                    colors.append('#ff4444')
                elif status == 'Distressed':
                    colors.append('#ffaa00')
                else:
                    colors.append('#00ff88')
            else:
                colors.append('#FFD700' if is_ccp else '#00ff88')  
            size = np.log10(max(row['total_assets'], 1e6)) * 2
            sizes.append(size)
            hover_texts.append(
                f"<b>{row['bank_name'][:40]}</b><br>"
                f"Region: {row['region']}<br>"
                f"Assets: ${row['total_assets']/1e9:.1f}B"
            )
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8,
            line=dict(width=1, color='#444')
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Banks'
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        showlegend=False
    )
    return fig
def create_equity_bar_chart(results):
    """Create horizontal bar chart of top losses."""
    if results is None:
        return go.Figure()
    df_results = pd.DataFrame({
        'bank': results.get('bank_names', [])[:10],
        'loss': (results['initial_equity'] - results['final_equity'])[:10] / 1e9
    })
    df_results = df_results.nlargest(10, 'loss')
    fig = go.Figure(go.Bar(
        x=df_results['loss'],
        y=df_results['bank'],
        orientation='h',
        marker_color='#ff4444'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Equity Loss ($B)",
        yaxis=dict(autorange="reversed")
    )
    return fig
def main():
    st.markdown("#  ENCS Systemic Risk Engine")
    st.markdown("### Eisenberg-Noe Contagion Simulation Dashboard")
    with st.spinner("Loading network data..."):
        W_dense_base, df_base = load_network_data()
    st.sidebar.markdown("---")
    st.sidebar.markdown("## \U0001f3e6 Market Structure")
    clearing_model = st.sidebar.radio(
        "Clearing Model",
        ["Bilateral (OTC)", "Central Cleared (CCP)"],
        index=0,
        help="Choose bilateral OTC or hub-and-spoke CCP topology"
    )
    if clearing_model == "Central Cleared (CCP)":
        clearing_rate = st.sidebar.slider(
            "Cleared Volume %", min_value=0, max_value=100, value=50,
            help="Percentage of interbank edges routed through the CCP"
        ) / 100.0
        default_fund_pct = st.sidebar.slider(
            "CCP Default Fund %", min_value=1, max_value=25, value=5,
            help="CCP equity as percentage of total cleared risk"
        ) / 100.0
        W_dense, df = sim.apply_central_clearing(
            W_dense_base, df_base, clearing_rate=clearing_rate,
            default_fund_ratio=default_fund_pct
        )
        st.sidebar.success(
            f"CCP active — {clearing_rate*100:.0f}% cleared, "
            f"{default_fund_pct*100:.0f}% default fund"
        )
    else:
        W_dense, df = W_dense_base, df_base
    initial_state = sim.compute_state_variables(W_dense, df)
    if 'network_coords' not in st.session_state or st.session_state.get('_ccp_mode') != clearing_model:
        coords, layout_df = generate_network_layout(df)
        st.session_state.network_coords = coords
        st.session_state.layout_df = layout_df
        st.session_state._ccp_mode = clearing_model
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    st.sidebar.markdown("##  Stress Test Controls")
    bank_options = df['bank_name'].tolist()
    selected_bank = st.sidebar.selectbox(
        " Select Trigger Bank",
        options=bank_options,
        index=0,
        help="Bank to shock (sorted by interbank exposure)"
    )
    trigger_idx = df[df['bank_name'] == selected_bank].index[0]
    severity = st.sidebar.slider(
        " Shock Severity",
        min_value=0,
        max_value=100,
        value=100,
        format="%d%%",
        help="Percentage of external assets destroyed"
    ) / 100.0
    with st.sidebar.expander(" Advanced Parameters"):
        max_iter = st.number_input("Max Iterations", value=100, min_value=10, max_value=500)
        tolerance = st.number_input("Convergence Tolerance", value=1e-5, format="%.0e")
        distress_thresh = st.slider("Distress Threshold", 0.0, 1.0, 0.95)
    st.sidebar.markdown("---")
    st.sidebar.markdown("##  Rust Intraday Engine")
    rust_badge = "\u26a1 Rust" if sim.RUST_AVAILABLE else "\U0001f40d Python fallback"
    st.sidebar.caption(f"Backend: {rust_badge}")
    intraday_mode = st.sidebar.checkbox("Enable Intraday Mode", value=False,
                                         help="Simulate discrete time steps with exponential fire sales")
    if intraday_mode:
        n_steps = st.sidebar.slider("Time Steps", min_value=1, max_value=50, value=10,
                                     help="Number of discrete intraday steps")
        intra_sigma = st.sidebar.slider("Market Uncertainty (\u03c3)",
                                         min_value=0.01, max_value=0.30, value=0.05, step=0.01,
                                         key="intra_sigma",
                                         help="Gaussian noise on solvency signals")
        intra_panic = st.sidebar.slider("Panic Threshold",
                                         min_value=0.0, max_value=0.50, value=0.10, step=0.01,
                                         key="intra_panic",
                                         help="Signal level below which creditors run")
        intra_alpha = st.sidebar.slider("Fire-Sale \u03b1 (Exponential Decay)",
                                         min_value=0.0, max_value=0.05, value=0.005, step=0.001,
                                         format="%.3f",
                                         help="P_new = P_old * exp(-\u03b1 * Volume)")
        intra_margin = st.sidebar.slider("Margin Sensitivity",
                                          min_value=0.0, max_value=5.0, value=1.0, step=0.1,
                                          help="Multiplier on derivative margin calls: MC_i = Deriv_i × (1-P) × Sensitivity")
    else:
        n_steps = 10
        intra_sigma = 0.05
        intra_panic = 0.10
        intra_alpha = 0.005
        intra_margin = 1.0
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("\U0001f680 RUN SIMULATION", use_container_width=True, type="primary")
    st.sidebar.markdown("---")
    st.sidebar.markdown("## \U0001f916 AI Risk Predictor")
    if not ML_AVAILABLE:
        st.sidebar.warning("torch / ml_pipeline not available")
    elif not GNN_MODEL_PATH.exists():
        st.sidebar.info("No trained model found. Run:\n`python ml_pipeline.py`")
    else:
        st.sidebar.caption("PNA model trained on Monte Carlo simulations")
        predict_button = st.sidebar.button("\U0001f52e PREDICT RISK", use_container_width=True)
        if predict_button:
            with st.spinner("Running GNN inference..."):
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = load_trained_model(str(GNN_MODEL_PATH), device=device)
                model = model.to(device)
                ai_pred = predict_risk(model, W_dense, df, device=device)
                st.session_state.ai_predictions = ai_pred
                st.sidebar.success(f"Predicted {(ai_pred['risk_labels']=='High Risk').sum()} high-risk nodes")
    if run_button:
        if intraday_mode:
            with st.spinner("Running Intraday Simulation..."):
                results = sim.run_rust_intraday(
                    initial_state, df, trigger_idx, severity,
                    n_steps=n_steps,
                    uncertainty_sigma=intra_sigma,
                    panic_threshold=intra_panic,
                    alpha=intra_alpha,
                    max_iterations=max_iter,
                    convergence_threshold=tolerance,
                    distress_threshold=distress_thresh,
                    margin_sensitivity=intra_margin
                )
                results['bank_names'] = df['bank_name'].tolist()
                st.session_state.simulation_results = results
        else:
            with st.spinner("Running Eisenberg-Noe clearing..."):
                results = sim.run_scenario(
                    initial_state, df, trigger_idx, severity,
                    max_iterations=max_iter,
                    convergence_threshold=tolerance,
                    distress_threshold=distress_thresh
                )
                results['bank_names'] = df['bank_name'].tolist()
                st.session_state.simulation_results = results
    results = st.session_state.simulation_results
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    if results:
        capital_vaporized = results['equity_loss'] / 1e12
        n_defaults = results['n_defaults']
        n_distressed = results['n_distressed']
        contagion = n_defaults + n_distressed - (1 if n_defaults > 0 else 0)
    else:
        capital_vaporized = 0
        n_defaults = 0
        n_distressed = 0
        contagion = 0
    with col1:
        st.metric(
            label=" Capital Vaporized",
            value=f"${capital_vaporized:.2f}T",
            delta=None
        )
    with col2:
        st.metric(
            label=" Defaults",
            value=n_defaults,
            delta=None
        )
    with col3:
        st.metric(
            label=" Distressed",
            value=n_distressed,
            delta=None
        )
    with col4:
        st.metric(
            label=" Contagion Count",
            value=contagion,
            delta=None
        )
    st.markdown("---")
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("###  3D Network Map")
        status_array = results['status'] if results else None
        fig = create_3d_network(
            df, 
            status_array=status_array,
            layout_df=st.session_state.layout_df,
            coords=st.session_state.network_coords
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        🟢 **Safe** | 🟠 **Distressed** | 🔴 **Default** | 🟡 **CCP**
        """)
    with col_right:
        st.markdown("###  Top 10 Casualties")
        if results:
            casualty_df = pd.DataFrame({
                'Bank': df['bank_name'].str[:35],
                'Region': df['region'],
                'Initial $B': (results['initial_equity'] / 1e9).round(1),
                'Final $B': (results['final_equity'] / 1e9).round(1),
                'Loss $B': ((results['initial_equity'] - results['final_equity']) / 1e9).round(1),
                'Status': results['status']
            })
            casualty_df = casualty_df.sort_values('Loss $B', ascending=False).head(10)
            def color_status(val):
                if val == 'Default':
                    return 'background-color: #ff4444'
                elif val == 'Distressed':
                    return 'background-color: #ffaa00'
                return ''
            st.dataframe(
                casualty_df.style.applymap(color_status, subset=['Status']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(" Click **RUN SIMULATION** to see results")
    if results:
        st.markdown("---")
        st.markdown("###  Simulation Details")
        col_det1, col_det2 = st.columns(2)
        with col_det1:
            st.markdown(f"""
            **Trigger Bank:** {results['trigger_name']}  
            **Shock Severity:** {results['loss_severity']*100:.0f}%  
            **Banks Analyzed:** {len(df):,}  
            """)
        with col_det2:
            if results['n_defaults'] > 0:
                defaults_us = sum(1 for i, s in enumerate(results['status']) 
                                 if s == 'Default' and df.iloc[i]['region'] == 'US')
                defaults_eu = results['n_defaults'] - defaults_us
                st.markdown(f"""
                **Defaults - US:** {defaults_us}  
                **Defaults - EU:** {defaults_eu}  
                """)
    if results and results.get('price_timeline'):
        st.markdown("---")
        st.markdown("### \u23f1 Intraday Contagion Timeline")
        tl_cols = st.columns(4)
        steps = list(range(1, len(results['price_timeline']) + 1))
        with tl_cols[0]:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=steps, y=results['price_timeline'],
                mode='lines+markers', name='Asset Price',
                line=dict(color='#ff4444', width=3),
                fill='tozeroy', fillcolor='rgba(255,68,68,0.15)'
            ))
            fig_price.update_layout(
                title="Asset Price (Exponential Decay)",
                xaxis_title="Time Step", yaxis_title="Price Multiplier",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=280, margin=dict(l=40, r=10, t=40, b=30),
                yaxis=dict(range=[0, 1.05])
            )
            st.plotly_chart(fig_price, use_container_width=True)
        with tl_cols[1]:
            fig_def = go.Figure()
            fig_def.add_trace(go.Bar(
                x=steps, y=results['defaults_timeline'],
                name='Defaults', marker_color='#ff4444'
            ))
            fig_def.add_trace(go.Bar(
                x=steps, y=results['distressed_timeline'],
                name='Distressed', marker_color='#ffaa00'
            ))
            fig_def.update_layout(
                title="Defaults & Distressed per Step",
                xaxis_title="Time Step", yaxis_title="Count",
                barmode='stack',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=280, margin=dict(l=40, r=10, t=40, b=30)
            )
            st.plotly_chart(fig_def, use_container_width=True)
        with tl_cols[2]:
            fig_grid = go.Figure()
            fig_grid.add_trace(go.Scatter(
                x=steps, y=results['gridlock_timeline'],
                mode='lines+markers', name='Failed Payments',
                line=dict(color='#ffaa00', width=3)
            ))
            fig_grid.update_layout(
                title="Liquidity Gridlock (Failed Payments)",
                xaxis_title="Time Step", yaxis_title="Failed Payments",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=280, margin=dict(l=40, r=10, t=40, b=30)
            )
            st.plotly_chart(fig_grid, use_container_width=True)
        with tl_cols[3]:
            margin_tl = results.get('margin_calls_timeline', [0] * len(steps))
            fig_margin = go.Figure()
            fig_margin.add_trace(go.Scatter(
                x=steps, y=[m / 1e9 for m in margin_tl],
                mode='lines+markers', name='Margin Calls',
                line=dict(color='#ff66ff', width=3),
                fill='tozeroy', fillcolor='rgba(255,102,255,0.15)'
            ))
            fig_margin.update_layout(
                title="Margin Call Spirals ($B)",
                xaxis_title="Time Step", yaxis_title="Margin Calls ($B)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=280, margin=dict(l=40, r=10, t=40, b=30)
            )
            st.plotly_chart(fig_margin, use_container_width=True)
    ai_pred = st.session_state.get('ai_predictions')
    if ai_pred is not None:
        st.markdown("---")
        st.markdown("### \U0001f916 AI Risk Prediction (GNN)")
        ai_col1, ai_col2 = st.columns([2, 1])
        with ai_col1:
            risk_probs = ai_pred['risk_probs']
            layout_df = st.session_state.layout_df
            coords = st.session_state.network_coords
            x_nodes, y_nodes, z_nodes = [], [], []
            colors_ai, sizes_ai, hover_ai = [], [], []
            for idx, row in layout_df.iterrows():
                if idx in coords:
                    x, y, z = coords[idx]
                    x_nodes.append(x)
                    y_nodes.append(y)
                    z_nodes.append(z)
                    p = risk_probs[idx] if idx < len(risk_probs) else 0.0
                    colors_ai.append(p)
                    size = np.log10(max(row['total_assets'], 1e6)) * 2
                    sizes_ai.append(size)
                    hover_ai.append(
                        f"<b>{row['bank_name'][:40]}</b><br>"
                        f"P(Risk): {p:.2%}<br>"
                        f"Region: {row['region']}<br>"
                        f"Assets: ${row['total_assets']/1e9:.1f}B"
                    )
            fig_ai = go.Figure()
            fig_ai.add_trace(go.Scatter3d(
                x=x_nodes, y=y_nodes, z=z_nodes,
                mode='markers',
                marker=dict(
                    size=sizes_ai,
                    color=colors_ai,
                    colorscale=[[0, '#00ff88'], [0.5, '#ffaa00'], [1.0, '#ff4444']],
                    cmin=0, cmax=1,
                    colorbar=dict(title='P(Risk)', tickformat='.0%'),
                    opacity=0.85,
                    line=dict(width=1, color='#444')
                ),
                text=hover_ai,
                hoverinfo='text',
                name='AI Risk'
            ))
            fig_ai.update_layout(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0),
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig_ai, use_container_width=True)
            st.markdown("\U0001f7e2 **Low Risk** → \U0001f7e0 **Medium** → \U0001f534 **High Risk** (continuous scale)")
        with ai_col2:
            st.markdown("#### Top 15 Riskiest Banks")
            top_n = 15
            sorted_idx = np.argsort(risk_probs)[::-1][:top_n]
            risk_df = pd.DataFrame({
                'Bank': [df.iloc[i]['bank_name'][:30] for i in sorted_idx],
                'Region': [df.iloc[i]['region'] for i in sorted_idx],
                'P(Risk)': [f"{risk_probs[i]:.1%}" for i in sorted_idx],
            })
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
            if results is not None:
                st.markdown("#### AI vs Simulation")
                sim_risky = np.isin(results['status'], ['Default', 'Distressed']).astype(int)
                ai_risky = (risk_probs >= 0.5).astype(int)
                min_len = min(len(sim_risky), len(ai_risky))
                sim_risky = sim_risky[:min_len]
                ai_risky = ai_risky[:min_len]
                tp = int(((ai_risky == 1) & (sim_risky == 1)).sum())
                tn = int(((ai_risky == 0) & (sim_risky == 0)).sum())
                fp = int(((ai_risky == 1) & (sim_risky == 0)).sum())
                fn = int(((ai_risky == 0) & (sim_risky == 1)).sum())
                accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                st.metric("Accuracy", f"{accuracy:.1%}")
                cm_col1, cm_col2 = st.columns(2)
                cm_col1.metric("Precision", f"{precision:.1%}")
                cm_col2.metric("Recall", f"{recall:.1%}")
                fig_cm = go.Figure(data=go.Heatmap(
                    z=[[tn, fp], [fn, tp]],
                    x=['Pred Safe', 'Pred Risky'],
                    y=['Actual Safe', 'Actual Risky'],
                    texttemplate='%{z:,}',
                    colorscale=[[0, '#1a1a2e'], [1, '#ff4444']],
                    showscale=False,
                ))
                fig_cm.update_layout(
                    title='Confusion Matrix',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=250,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Run a simulation to compare AI vs actual results")
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🌍 Climate Stress Test")
    st.sidebar.caption("Green Swan scenario — sudden carbon tax strands brown assets")
    climate_enabled = st.sidebar.checkbox(
        "Enable Green Swan Scenario", value=False, key="climate_on",
        help="Apply a carbon-tax transition shock based on portfolio composition"
    )
    if climate_enabled:
        climate_tax = st.sidebar.slider(
            "Carbon Tax Severity", min_value=0, max_value=100, value=50,
            format="%d%%", key="climate_tax",
            help="Percentage of brown (fossil-fuel) assets destroyed"
        ) / 100.0
        climate_subsidy = st.sidebar.slider(
            "Green Subsidy", min_value=0, max_value=50, value=10,
            format="%d%%", key="climate_sub",
            help="Percentage gain on green (renewables) assets"
        ) / 100.0
        climate_intraday = st.sidebar.checkbox(
            "Use Intraday Engine", value=True, key="climate_intra",
            help="Propagate via intraday fire-sale engine (unchecked = Eisenberg-Noe)"
        )
        run_climate_btn = st.sidebar.button(
            "🌍 RUN CLIMATE SHOCK", use_container_width=True, key="run_climate"
        )
        if run_climate_btn:
            with st.spinner("Running Green Swan scenario…"):
                df_climate = assign_climate_exposure(df.copy())
                climate_state = sim.compute_state_variables(W_dense, df_climate)
                climate_results = run_transition_shock(
                    climate_state, df_climate,
                    carbon_tax_severity=climate_tax,
                    green_subsidy=climate_subsidy,
                    use_intraday=climate_intraday,
                    n_steps=n_steps,
                    uncertainty_sigma=intra_sigma,
                    panic_threshold=intra_panic,
                    alpha=intra_alpha,
                    margin_sensitivity=intra_margin,
                    max_iterations=int(max_iter),
                    convergence_threshold=tolerance,
                    distress_threshold=distress_thresh,
                )
                st.session_state.climate_results = climate_results
                st.session_state.df_climate = df_climate
            st.sidebar.success("Climate shock complete — scroll down for results ↓")
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ♟️ Strategic Simulation")
    st.sidebar.caption("Morris & Shin (1998) Global Games — coordination failure model")
    with st.sidebar.expander("Game Parameters", expanded=False):
        game_n_banks = st.number_input("Number of agents", value=20, min_value=5, max_value=100, key="game_n_banks")
        game_n_steps = st.number_input("Time steps", value=5, min_value=2, max_value=20, key="game_n_steps")
        game_solvency = st.slider("True solvency (θ)", -0.05, 0.30, 0.20, 0.01, key="game_solv")
        game_rate = st.slider("Interest rate (r)", 0.01, 0.20, 0.10, 0.01, key="game_rate")
        game_recovery = st.slider("Recovery rate (R)", 0.10, 0.80, 0.40, 0.05, key="game_rec")
        game_lambda = st.slider("Risk aversion (λ mean)", 0.1, 3.0, 1.0, 0.1, key="game_lam")
        game_noise = st.slider("Private noise (σ)", 0.01, 0.30, 0.08, 0.01, key="game_noise")
        game_haircut = st.slider("Fire-sale haircut", 0.05, 0.50, 0.20, 0.05, key="game_hc")
        game_margin = st.slider("Margin volatility", 0.0, 1.0, 0.3, 0.1, key="game_mv")
        game_exposure = st.number_input("Exposure / bank ($B)", value=1.0, min_value=0.1, max_value=50.0, step=0.5, key="game_exp")
    run_game_btn = st.sidebar.button("♟️ RUN GAME A/B TEST", use_container_width=True)
    if run_game_btn:
        with st.spinner("Running Global Games A/B test…"):
            common = dict(
                n_banks=int(game_n_banks),
                n_steps=int(game_n_steps),
                true_solvency=game_solvency,
                interest_rate=game_rate,
                recovery_rate=game_recovery,
                risk_aversion_mean=game_lambda,
                private_noise_std=game_noise,
                initial_exposure_per_bank=game_exposure * 1e9,
                fire_sale_haircut=game_haircut,
                margin_volatility=game_margin,
                seed=42,
            )
            res_opaque = run_game_simulation(info_regime="OPAQUE", **common)
            res_transparent = run_game_simulation(info_regime="TRANSPARENT", **common)
            st.session_state.game_opaque = res_opaque
            st.session_state.game_transparent = res_transparent
        st.sidebar.success("Game A/B test complete — scroll down for results ↓")
    if st.session_state.get('game_opaque') and st.session_state.get('game_transparent'):
        res_a = st.session_state.game_opaque
        res_b = st.session_state.game_transparent
        tl_a = res_a['timeline']
        tl_b = res_b['timeline']
        loss_a = res_a['total_fire_sale_loss']
        loss_b = res_b['total_fire_sale_loss']
        capital_saved = loss_a - loss_b
        st.markdown("---")
        st.markdown("### ♟️ Strategic Simulation — Global Games A/B Test")
        st.markdown(
            "_Morris & Shin (1998): banks fail from **coordination failure** "
            "(panics), not just insolvency.  "
            "The AI transparency signal anchors expectations and prevents self-fulfilling runs._"
        )
        gcol1, gcol2, gcol3 = st.columns(3)
        with gcol1:
            st.metric("🔴 Loss — Opaque (Fog of War)", f"${loss_a / 1e9:,.2f}B")
        with gcol2:
            st.metric("🟢 Loss — Transparent (AI)", f"${loss_b / 1e9:,.2f}B")
        with gcol3:
            st.markdown(
                f"<div style='text-align:center; padding:12px; "
                f"border:2px solid #00ff88; border-radius:10px;'>"
                f"<span style='font-size:0.9rem;color:#888;'>Capital Saved by AI</span><br/>"
                f"<span style='font-size:2.5rem;font-weight:bold;color:#00ff88;'>"
                f"${capital_saved / 1e9:,.2f}B</span></div>",
                unsafe_allow_html=True,
            )
        fig_game = go.Figure()
        fig_game.add_trace(go.Scatter(
            x=tl_a['steps'],
            y=[v / 1e9 for v in tl_a['cumulative_fire_sale_loss']],
            mode='lines+markers',
            name='Opaque (Fog of War)',
            line=dict(color='#ff4444', width=3),
            fill='tozeroy', fillcolor='rgba(255,68,68,0.12)',
        ))
        fig_game.add_trace(go.Scatter(
            x=tl_b['steps'],
            y=[v / 1e9 for v in tl_b['cumulative_fire_sale_loss']],
            mode='lines+markers',
            name='Transparent (AI Signal)',
            line=dict(color='#00ff88', width=3),
            fill='tozeroy', fillcolor='rgba(0,255,136,0.12)',
        ))
        fig_game.update_layout(
            title='Cumulative Fire-Sale Losses Over Time',
            xaxis_title='Time Step', yaxis_title='Cumulative Loss ($B)',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=370, margin=dict(l=50, r=20, t=50, b=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        st.plotly_chart(fig_game, use_container_width=True)
        gc1, gc2 = st.columns(2)
        with gc1:
            fig_belief = go.Figure()
            fig_belief.add_trace(go.Scatter(
                x=tl_a['steps'], y=tl_a['avg_belief'],
                mode='lines+markers', name='Opaque',
                line=dict(color='#ff4444', width=2),
            ))
            fig_belief.add_trace(go.Scatter(
                x=tl_b['steps'], y=tl_b['avg_belief'],
                mode='lines+markers', name='Transparent',
                line=dict(color='#00ff88', width=2),
            ))
            fig_belief.update_layout(
                title='Average P(Default) Belief',
                xaxis_title='Step', yaxis_title='P(default)',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=280, margin=dict(l=40, r=10, t=40, b=30),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_belief, use_container_width=True)
        with gc2:
            fig_run = go.Figure()
            fig_run.add_trace(go.Bar(
                x=tl_a['steps'], y=[r * 100 for r in tl_a['run_fraction']],
                name='Opaque', marker_color='#ff4444',
            ))
            fig_run.add_trace(go.Bar(
                x=tl_b['steps'], y=[r * 100 for r in tl_b['run_fraction']],
                name='Transparent', marker_color='#00ff88',
            ))
            fig_run.update_layout(
                title='Withdrawal Rate per Step (%)',
                xaxis_title='Step', yaxis_title='% Agents Withdrawing',
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=280, margin=dict(l=40, r=10, t=40, b=30),
                yaxis=dict(range=[0, 105]),
            )
            st.plotly_chart(fig_run, use_container_width=True)
        with st.expander("Step-by-step detail", expanded=False):
            detail_a = pd.DataFrame({
                'Step': tl_a['steps'],
                'Runs (A)': tl_a['n_runs'],
                'Run% (A)': [f"{x:.0%}" for x in tl_a['run_fraction']],
                'P(def) A': [f"{x:.4f}" for x in tl_a['avg_belief']],
                'Loss A ($B)': [f"{x/1e9:.2f}" for x in tl_a['cumulative_fire_sale_loss']],
                'Runs (B)': tl_b['n_runs'],
                'Run% (B)': [f"{x:.0%}" for x in tl_b['run_fraction']],
                'P(def) B': [f"{x:.4f}" for x in tl_b['avg_belief']],
                'Loss B ($B)': [f"{x/1e9:.2f}" for x in tl_b['cumulative_fire_sale_loss']],
            })
            st.dataframe(detail_a, use_container_width=True, hide_index=True)
    if st.session_state.get('climate_results'):
        cr = st.session_state.climate_results
        df_c = st.session_state.df_climate
        st.markdown("---")
        st.markdown("### 🌍 Green Swan — Climate Transition Risk")
        st.markdown(
            "_A sudden carbon tax strands **Brown Assets** (fossil fuels). "
            "US banks (high carbon exposure) absorb the direct hit, then "
            "transmit losses to EU banks via interbank leverage — proving "
            "climate risk is **systemic**._"
        )
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            st.metric(
                "🏭 Brown Losses",
                f"${cr['total_brown_loss'] / 1e9:,.1f}B",
            )
        with cc2:
            st.metric(
                "🌱 Green Gains",
                f"${cr['total_green_gain'] / 1e9:,.1f}B",
            )
        with cc3:
            st.metric(
                "💀 Capital Destroyed",
                f"${cr['equity_loss'] / 1e9:,.1f}B",
            )
        with cc4:
            st.metric(
                "🏦 Bank Failures",
                cr['n_defaults'],
            )
        st.write(
            "**Observe:** high-carbon US banks fail first, transmitting the shock "
            "to low-carbon EU banks through interbank leverage. Even 'green' banks "
            "are dragged into distress — climate risk is a *network* phenomenon."
        )
        status_arr = np.array(cr['status'])  
        regions_arr = df_c['region'].values
        breakdown = []
        for rgn in ['US', 'EU']:
            rmask = (regions_arr == rgn)
            breakdown.append({
                'Region': rgn,
                'Defaults': int(((status_arr == 'Default') & rmask).sum()),
                'Distressed': int(((status_arr == 'Distressed') & rmask).sum()),
                'Avg Carbon Score': f"{df_c.loc[rmask, 'carbon_score'].mean():.2f}",
                'Brown Losses ($B)': f"{cr['climate_losses'][rmask].sum() / 1e9:,.1f}",
                'Green Gains ($B)': f"{cr['climate_gains'][rmask].sum() / 1e9:,.1f}",
            })
        st.dataframe(
            pd.DataFrame(breakdown),
            use_container_width=True, hide_index=True,
        )
        clim_col1, clim_col2 = st.columns(2)
        with clim_col1:
            top_n = 15
            net_shock = cr['climate_net_shock']
            worst_idx = np.argsort(net_shock)[::-1][:top_n]
            casualty = pd.DataFrame({
                'Bank': [str(df_c.iloc[i]['bank_name'])[:30] for i in worst_idx],
                'Region': [df_c.iloc[i]['region'] for i in worst_idx],
                'Carbon': [f"{df_c.iloc[i]['carbon_score']:.2f}" for i in worst_idx],
                'Net Hit ($B)': [f"{net_shock[i] / 1e9:,.1f}" for i in worst_idx],
                'Status': [status_arr[i] for i in worst_idx],
            })
            st.markdown("#### Top 15 Climate Casualties")
            st.dataframe(casualty, use_container_width=True, hide_index=True)
        with clim_col2:
            fig_cs = go.Figure()
            color_map = {'Default': '#ff4444', 'Distressed': '#ffaa00', 'Safe': '#00ff88'}
            for status_val, color in color_map.items():
                mask = (status_arr == status_val)
                if not mask.any():
                    continue
                fig_cs.add_trace(go.Scatter(
                    x=df_c.loc[mask, 'carbon_score'],
                    y=cr['climate_net_shock'][mask] / 1e9,
                    mode='markers',
                    name=status_val,
                    marker=dict(
                        color=color, size=8, opacity=0.7,
                        line=dict(width=0.5, color='#333'),
                    ),
                    text=df_c.loc[mask, 'bank_name'].str[:25],
                    hovertemplate='%{text}<br>CS: %{x:.2f}<br>Net Hit: $%{y:.1f}B<extra></extra>',
                ))
            fig_cs.update_layout(
                title='Carbon Score vs Net Climate Shock',
                xaxis_title='Carbon Score (0=Green, 1=Brown)',
                yaxis_title='Net Climate Shock ($B)',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=380, margin=dict(l=50, r=20, t=50, b=40),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            )
            st.plotly_chart(fig_cs, use_container_width=True)
        if cr.get('price_timeline'):
            steps = list(range(1, len(cr['price_timeline']) + 1))
            tc1, tc2 = st.columns(2)
            with tc1:
                fig_cp = go.Figure()
                fig_cp.add_trace(go.Scatter(
                    x=steps, y=cr['price_timeline'],
                    mode='lines+markers', name='Asset Price',
                    line=dict(color='#ff4444', width=3),
                    fill='tozeroy', fillcolor='rgba(255,68,68,0.12)',
                ))
                fig_cp.update_layout(
                    title='Asset Price During Climate Cascade',
                    xaxis_title='Step', yaxis_title='Price Multiplier',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    height=280, margin=dict(l=40, r=10, t=40, b=30),
                    yaxis=dict(range=[0, 1.05]),
                )
                st.plotly_chart(fig_cp, use_container_width=True)
            with tc2:
                fig_cd = go.Figure()
                fig_cd.add_trace(go.Bar(
                    x=steps, y=cr['defaults_timeline'],
                    name='Defaults', marker_color='#ff4444',
                ))
                fig_cd.add_trace(go.Bar(
                    x=steps, y=cr['distressed_timeline'],
                    name='Distressed', marker_color='#ffaa00',
                ))
                fig_cd.update_layout(
                    title='Climate-Induced Failures per Step',
                    xaxis_title='Step', yaxis_title='Count', barmode='stack',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    height=280, margin=dict(l=40, r=10, t=40, b=30),
                )
                st.plotly_chart(fig_cd, use_container_width=True)
    st.markdown("---")
    st.caption("ENCS Systemic Risk Engine | Hybrid Rust/Python Architecture | Eisenberg-Noe + Intraday Fire Sales + GNN Risk Predictor + Global Games + Green Swan")
if __name__ == "__main__":
    main()