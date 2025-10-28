"""Visualization utilities for predictions."""
import plotly.graph_objects as go
from typing import Dict


def create_prediction_plot(predictions: Dict[str, float], model_name: str, top_k: int = 10) -> go.Figure:
    """
    Create a Plotly bar chart for model predictions.
    
    Args:
        predictions: Dictionary mapping labels to probabilities
        model_name: Name of the model (for title)
        top_k: Number of top predictions to show
        
    Returns:
        Plotly Figure object
    """
    # Sort predictions by probability (descending)
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Take top_k predictions
    top_preds = sorted_preds[:top_k]
    
    # Extract labels and probabilities
    labels = [pred[0] for pred in top_preds]
    probs = [pred[1] for pred in top_preds]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=labels,
            orientation='h',
            marker=dict(
                color=probs,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Probability")
            ),
            text=[f"{p:.3f}" for p in probs],
            textposition='auto',
        )
    ])
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{model_name.title()} Model - Top {top_k} Predictions",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Probability",
        yaxis_title="Label",
        yaxis=dict(autorange="reversed"),  # Highest probability at top
        height=max(400, top_k * 40),  # Dynamic height based on number of bars
        margin=dict(l=150, r=50, t=80, b=50),
        font=dict(size=12),
        plot_bgcolor='rgba(240,240,240,0.5)',
    )
    
    return fig


def create_combined_plot(action_preds: Dict[str, float], bodyparts_preds: Dict[str, float], top_k: int = 10) -> go.Figure:
    """
    Create a combined plot showing predictions from both models side by side.
    
    Args:
        action_preds: Predictions from action model
        bodyparts_preds: Predictions from bodyparts model
        top_k: Number of top predictions to show per model
        
    Returns:
        Plotly Figure with subplots
    """
    from plotly.subplots import make_subplots
    
    # Sort and get top predictions for both models
    action_sorted = sorted(action_preds.items(), key=lambda x: x[1], reverse=True)[:top_k]
    bodyparts_sorted = sorted(bodyparts_preds.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    action_labels = [pred[0] for pred in action_sorted]
    action_probs = [pred[1] for pred in action_sorted]
    
    bodyparts_labels = [pred[0] for pred in bodyparts_sorted]
    bodyparts_probs = [pred[1] for pred in bodyparts_sorted]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Action Predictions", "Bodyparts Predictions"),
        horizontal_spacing=0.15
    )
    
    # Add action predictions
    fig.add_trace(
        go.Bar(
            x=action_probs,
            y=action_labels,
            orientation='h',
            marker=dict(color='#FF6B6B'),
            text=[f"{p:.3f}" for p in action_probs],
            textposition='auto',
            name='Action'
        ),
        row=1, col=1
    )
    
    # Add bodyparts predictions
    fig.add_trace(
        go.Bar(
            x=bodyparts_probs,
            y=bodyparts_labels,
            orientation='h',
            marker=dict(color='#4ECDC4'),
            text=[f"{p:.3f}" for p in bodyparts_probs],
            textposition='auto',
            name='Bodyparts'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Model Predictions Comparison",
            x=0.5,
            xanchor='center',
            font=dict(size=18, family='Arial Black')
        ),
        showlegend=False,
        height=max(500, top_k * 45),
        font=dict(size=11),
        plot_bgcolor='rgba(240,240,240,0.5)',
    )
    
    # Update axes
    fig.update_xaxes(title_text="Probability", row=1, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Probability", row=1, col=2, range=[0, 1])
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    
    return fig

