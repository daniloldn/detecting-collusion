import plotly.graph_objects as go
import pandas as pd

def plot_market_plotly(df: pd.DataFrame, market_id: int = 0):
    d = df[df["market_id"] == market_id].copy()
    d = d.sort_values("t")

    fig = go.Figure()

    # Log price line
    fig.add_trace(
        go.Scatter(
            x=d["t"],
            y=d["p"],
            mode="lines",
            name="Log Price",
            line=dict(color="black")
        )
    )

    fig.add_trace(
        go.Scatter(
            x = d["t"],
            y = d["c"],
            mode = "lines",
            name="Cost",
            line=dict(color="blue")

        )
    )

    # Add shaded regions for regimes
    regime_colors = {
        0: "rgba(0,200,0,0.15)",   # Competitive (green)
        1: "rgba(255,165,0,0.15)", # Tacit (orange)
        2: "rgba(200,0,0,0.15)"    # Cartel (red)
    }

    current_state = d["S"].iloc[0]
    start_t = d["t"].iloc[0]

    for i in range(1, len(d)):
        if d["S"].iloc[i] != current_state:
            end_t = d["t"].iloc[i]
            fig.add_vrect(
                x0=start_t,
                x1=end_t,
                fillcolor=regime_colors[current_state],
                line_width=0,
            )
            current_state = d["S"].iloc[i]
            start_t = d["t"].iloc[i]

    # last segment
    fig.add_vrect(
        x0=start_t,
        x1=d["t"].iloc[-1],
        fillcolor=regime_colors[current_state],
        line_width=0,
    )

    fig.update_layout(
        title=f"Market {market_id} — Log Price with Regime Shading",
        xaxis_title="Month",
        yaxis_title="Log Price",
        template="plotly_white"
    )

    fig.show()