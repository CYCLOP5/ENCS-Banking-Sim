import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import simulation_engine as sim
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
            if status_array is not None and idx < len(status_array):
                status = status_array[idx]
                if status == 'Default':
                    colors.append('#ff4444')
                elif status == 'Distressed':
                    colors.append('#ffaa00')
                else:
                    colors.append('#00ff88')
            else:
                colors.append('#00ff88')  
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
        W_dense, df = load_network_data()
        initial_state = compute_initial_state(W_dense, df)
    if 'network_coords' not in st.session_state:
        coords, layout_df = generate_network_layout(df)
        st.session_state.network_coords = coords
        st.session_state.layout_df = layout_df
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
    st.sidebar.markdown("## \U0001F9E0 Rust Intraday Engine")
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
    else:
        n_steps = 10
        intra_sigma = 0.05
        intra_panic = 0.10
        intra_alpha = 0.005

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("\U0001f680 RUN SIMULATION", use_container_width=True, type="primary")
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
                    distress_threshold=distress_thresh
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
        🟢 **Safe** | 🟠 **Distressed** | 🔴 **Default**
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
        tl_cols = st.columns(3)

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

    st.markdown("---")
    st.caption("ENCS Systemic Risk Engine | Hybrid Rust/Python Architecture | Eisenberg-Noe + Intraday Fire Sales")
if __name__ == "__main__":
    main()