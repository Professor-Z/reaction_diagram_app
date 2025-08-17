import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json

# Set page configuration
st.set_page_config(
    page_title="Reaction Coordinate Diagram",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state variables
if 'reactant_energy' not in st.session_state:
    st.session_state.reactant_energy = 10
if 'product_energy' not in st.session_state:
    st.session_state.product_energy = 10
if 'peak_y' not in st.session_state:
    st.session_state.peak_y = 11.4
if 'peak_x' not in st.session_state:
    st.session_state.peak_x = 5
if 'show_ea' not in st.session_state:
    st.session_state.show_ea = False
if 'show_deltaH' not in st.session_state:
    st.session_state.show_deltaH = False
if 'show_labels' not in st.session_state:
    st.session_state.show_labels = True
if 'endothermic_mode' not in st.session_state:
    st.session_state.endothermic_mode = False
if 'exothermic_mode' not in st.session_state:
    st.session_state.exothermic_mode = False
if 'saved_curves' not in st.session_state:
    st.session_state.saved_curves = []

# Constants
ENDOTHERMIC_REACTANT_ENERGY = 10
ENDOTHERMIC_PRODUCT_ENERGY = 12
ENDOTHERMIC_PEAK_ENERGY = 13.55
EXOTHERMIC_REACTANT_ENERGY = 10
EXOTHERMIC_PRODUCT_ENERGY = 8
EXOTHERMIC_PEAK_ENERGY = 11.4

# Define color spectrum for ΔH mapping
color_list = [
    "#9000FF",  # purple
    "#FB00FF",  # purple
    "#002AFF",  # blue    
    "#00BBFF",  # blue
    "#037303",  # green    
    "#03E203",  # green
    '#FFFF00',  # yellow
    "#F6B132",  # orange
    '#FF0000',  # red
    "#6A0202"   # dark red
]

# ΔH range
deltaH_min = -5
deltaH_max = 5

def get_color_for_deltaH(deltaH):
    """Map ΔH value to color from the spectrum"""
    deltaH_clamped = np.clip(deltaH, deltaH_min, deltaH_max)
    t = (deltaH_clamped - deltaH_min) / (deltaH_max - deltaH_min)
    
    # Interpolate through color list
    if t <= 0:
        return color_list[0]
    elif t >= 1:
        return color_list[-1]
    else:
        # Find position in color list
        pos = t * (len(color_list) - 1)
        idx = int(pos)
        frac = pos - idx
        
        if idx >= len(color_list) - 1:
            return color_list[-1]
        
        # Simple color interpolation (hex to RGB and back)
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        
        rgb1 = hex_to_rgb(color_list[idx])
        rgb2 = hex_to_rgb(color_list[idx + 1])
        
        interpolated_rgb = tuple(rgb1[i] + frac * (rgb2[i] - rgb1[i]) for i in range(3))
        return rgb_to_hex(interpolated_rgb)

def get_transition_color(peak_x):
    """Get transition state color based on x-position"""
    t = (peak_x - 1) / (9 - 1)
    t = np.clip(t, 0, 1)
    
    if t <= 0.5:
        t2 = t / 0.5
        r = 0 + t2 * 0.5
        g = 0
        b = 1 + t2 * (-0.5)
    else:
        t2 = (t - 0.5) / 0.5
        r = 0.5 + t2 * 0.5
        g = 0
        b = 0.5 + t2 * (-0.5)
    
    # Convert to hex
    r_hex = format(int(r * 255), '02x')
    g_hex = format(int(g * 255), '02x')
    b_hex = format(int(b * 255), '02x')
    
    return f'#{r_hex}{g_hex}{b_hex}'

def cubic_with_conditions(x0, y0, x1, y1, x2, y2, slope0=0, slope2=0):
    """Generate cubic polynomial with boundary conditions"""
    A_full = np.array([
        [x0**3, x0**2, x0, 1],
        [3*x0**2, 2*x0, 1, 0],
        [x2**3, x2**2, x2, 1],
        [3*x2**2, 2*x2, 1, 0],
    ])
    b_full = np.array([y0, slope0, y2, slope2])
    coeffs = np.linalg.solve(A_full, b_full)
    return np.poly1d(coeffs)

def compute_energy():
    """Compute the energy profile curve"""
    x = np.linspace(0, 10, 500)
    
    # Left segment (reactant to TS)
    x_inflect_left = st.session_state.peak_x / 2
    y_inflect_left = (st.session_state.reactant_energy + st.session_state.peak_y) / 2
    left_poly = cubic_with_conditions(
        x0=0, y0=st.session_state.reactant_energy,
        x1=x_inflect_left, y1=y_inflect_left,
        x2=st.session_state.peak_x, y2=st.session_state.peak_y,
        slope0=0, slope2=0
    )
    
    # Right segment (TS to product)
    x_inflect_right = st.session_state.peak_x + (10 - st.session_state.peak_x) / 2
    y_inflect_right = (st.session_state.peak_y + st.session_state.product_energy) / 2
    right_poly = cubic_with_conditions(
        x0=st.session_state.peak_x, y0=st.session_state.peak_y,
        x1=x_inflect_right, y1=y_inflect_right,
        x2=10, y2=st.session_state.product_energy,
        slope0=0, slope2=0
    )
    
    # Combine segments
    y_vals = np.empty_like(x)
    mask_left = x <= st.session_state.peak_x
    mask_right = x > st.session_state.peak_x
    y_vals[mask_left] = left_poly(x[mask_left])
    y_vals[mask_right] = right_poly(x[mask_right])
    
    return x, y_vals

def update_hammond_postulate():
    """Update transition state position based on Hammond's postulate"""
    if st.session_state.endothermic_mode:
        st.session_state.reactant_energy = ENDOTHERMIC_REACTANT_ENERGY
    elif st.session_state.exothermic_mode:
        st.session_state.reactant_energy = EXOTHERMIC_REACTANT_ENERGY
    
    # Compute ΔH for Hammond's postulate
    delta_H = st.session_state.product_energy - st.session_state.reactant_energy
    
    # Map ΔH to normalized value [-1, 1], clamp
    max_delta = 10
    delta_clamped = np.clip(delta_H, -max_delta, max_delta)
    norm_delta = delta_clamped / max_delta
    
    # Map to position between 1 and 9
    st.session_state.peak_x = 5 + norm_delta * 4
    st.session_state.peak_x = np.clip(st.session_state.peak_x, 1, 9)

def create_interactive_plot():
    """Create the interactive reaction coordinate plot with draggable points"""
    x, y = compute_energy()
    
    # Create the main plot
    fig = go.Figure()
    
    # Add saved curves first (so they appear behind)
    for i, curve_data in enumerate(st.session_state.saved_curves):
        fig.add_trace(go.Scatter(
            x=curve_data['x'],
            y=curve_data['y'],
            mode='lines',
            line=dict(color=curve_data['color'], width=2),
            name=f'Saved Curve {i+1}',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Main energy curve
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='grey', width=3),
        name='Energy Profile',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Draggable points and labels
    if st.session_state.show_labels:
        # Horizontal dashed lines
        fig.add_hline(y=st.session_state.reactant_energy, 
                     line_dash="dash", line_color="grey", opacity=0.5)
        fig.add_hline(y=st.session_state.product_energy, 
                     line_dash="dash", line_color="grey", opacity=0.5)
        fig.add_hline(y=st.session_state.peak_y, 
                     line_dash="dash", line_color="grey", opacity=0.5)
        
        # Reactant point (draggable in y-direction only)
        fig.add_trace(go.Scatter(
            x=[0], y=[st.session_state.reactant_energy],
            mode='markers+text',
            marker=dict(size=15, color='blue', line=dict(width=2, color='darkblue')),
            text=['REACTANT'],
            textposition='top center',
            textfont=dict(size=12, color='blue'),
            name='Reactant',
            showlegend=False,
            customdata=['reactant'],
            hovertemplate='<b>Reactant</b><br>Energy: %{y:.1f}<extra></extra>'
        ))
        
        # Product point (draggable in y-direction only)
        fig.add_trace(go.Scatter(
            x=[10], y=[st.session_state.product_energy],
            mode='markers+text',
            marker=dict(size=15, color='red', line=dict(width=2, color='darkred')),
            text=['PRODUCT'],
            textposition='bottom center',
            textfont=dict(size=12, color='red'),
            name='Product',
            showlegend=False,
            customdata=['product'],
            hovertemplate='<b>Product</b><br>Energy: %{y:.1f}<extra></extra>'
        ))
        
        # Transition state point (draggable in y-direction only)
        ts_color = get_transition_color(st.session_state.peak_x)
        fig.add_trace(go.Scatter(
            x=[st.session_state.peak_x], y=[st.session_state.peak_y],
            mode='markers+text',
            marker=dict(size=15, color=ts_color, line=dict(width=2, color='purple')),
            text=['TRANSITION STATE'],
            textposition='top center',
            textfont=dict(size=12, color=ts_color),
            name='Transition State',
            showlegend=False,
            customdata=['transition_state'],
            hovertemplate='<b>Transition State</b><br>Energy: %{y:.1f}<br>Position: %{x:.1f}<extra></extra>'
        ))
    
    # Add Ea arrow if enabled
    if st.session_state.show_ea:
        x_arrow = st.session_state.peak_x - 0.5
        fig.add_annotation(
            x=x_arrow, y=(st.session_state.reactant_energy + st.session_state.peak_y) / 2,
            text="E<sub>a</sub>",
            showarrow=False,
            font=dict(color='green', size=16),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='green',
            borderwidth=1
        )
        
        # Vertical arrow line
        fig.add_shape(
            type="line",
            x0=x_arrow, y0=st.session_state.reactant_energy,
            x1=x_arrow, y1=st.session_state.peak_y,
            line=dict(color="green", width=2),
        )
        
        # Arrow heads
        fig.add_annotation(
            x=x_arrow, y=st.session_state.reactant_energy + 0.2,
            ax=0, ay=10, arrowhead=2, arrowcolor="green", arrowwidth=2,
            showarrow=True, text=""
        )
        fig.add_annotation(
            x=x_arrow, y=st.session_state.peak_y - 0.2,
            ax=0, ay=-10, arrowhead=2, arrowcolor="green", arrowwidth=2,
            showarrow=True, text=""
        )
    
    # Add ΔH arrow if enabled
    if st.session_state.show_deltaH:
        x_deltaH = st.session_state.peak_x + 0.5
        fig.add_annotation(
            x=x_deltaH, y=(st.session_state.reactant_energy + st.session_state.product_energy) / 2,
            text="ΔH",
            showarrow=False,
            font=dict(color='orange', size=16),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='orange',
            borderwidth=1
        )
        
        # Vertical arrow line
        fig.add_shape(
            type="line",
            x0=x_deltaH, y0=st.session_state.reactant_energy,
            x1=x_deltaH, y1=st.session_state.product_energy,
            line=dict(color="orange", width=2),
        )
        
        # Arrow heads
        min_y = min(st.session_state.reactant_energy, st.session_state.product_energy)
        max_y = max(st.session_state.reactant_energy, st.session_state.product_energy)
        
        fig.add_annotation(
            x=x_deltaH, y=min_y + 0.2,
            ax=0, ay=10, arrowhead=2, arrowcolor="orange", arrowwidth=2,
            showarrow=True, text=""
        )
        fig.add_annotation(
            x=x_deltaH, y=max_y - 0.2,
            ax=0, ay=-10, arrowhead=2, arrowcolor="orange", arrowwidth=2,
            showarrow=True, text=""
        )
    
    # Configure layout
    fig.update_layout(
        title=dict(
            text="Interactive Reaction Coordinate Diagram",
            font=dict(size=24),
            x=0.5
        ),
        xaxis_title=dict(text="Reaction Coordinate", font=dict(size=16)),
        yaxis_title=dict(text="Energy", font=dict(size=16)),
        xaxis=dict(
            range=[-1, 11],
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            dtick=1,
            showticklabels=False  # Hide tick labels like original
        ),
        yaxis=dict(
            range=[4, 18],
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            dtick=1,
            showticklabels=False  # Hide tick labels like original
        ),
        plot_bgcolor='white',
        height=700,
        margin=dict(t=100, b=100, l=100, r=100),
        dragmode=False,  # Disable pan mode
        hovermode='closest'
    )
    
    return fig

# Main app
st.title("⚗️ Interactive Reaction Coordinate Diagram")

# Instructions
st.markdown("""
**Instructions:** Adjust the energy values using the sliders below, then view the reaction coordinate diagram. 
Use the control buttons below the graph to toggle different features and save curves for comparison.
""")

# Energy control sliders at the top
st.markdown("### Energy Controls")

# Create three columns for sliders
slider_col1, slider_col2, slider_col3 = st.columns(3)

with slider_col1:
    if not (st.session_state.endothermic_mode or st.session_state.exothermic_mode):
        new_reactant = st.slider(
            "Reactant Energy",
            min_value=4.0,
            max_value=18.0,
            value=float(st.session_state.reactant_energy),
            step=0.1,
            key="reactant_slider"
        )
        if new_reactant != st.session_state.reactant_energy:
            st.session_state.reactant_energy = new_reactant
            update_hammond_postulate()
            st.rerun()
    else:
        st.write(f"Reactant Energy: {st.session_state.reactant_energy:.1f} (Fixed in mode)")

with slider_col2:
    if st.session_state.endothermic_mode:
        new_product = st.slider(
            "Product Energy",
            min_value=float(st.session_state.reactant_energy),
            max_value=18.0,
            value=float(st.session_state.product_energy),
            step=0.1,
            key="product_slider_endo"
        )
    elif st.session_state.exothermic_mode:
        new_product = st.slider(
            "Product Energy",
            min_value=4.0,
            max_value=float(st.session_state.reactant_energy),
            value=float(st.session_state.product_energy),
            step=0.1,
            key="product_slider_exo"
        )
    else:
        new_product = st.slider(
            "Product Energy",
            min_value=4.0,
            max_value=18.0,
            value=float(st.session_state.product_energy),
            step=0.1,
            key="product_slider_default"
        )
    
    if new_product != st.session_state.product_energy:
        st.session_state.product_energy = new_product
        
        # Update peak energy based on mode
        if st.session_state.endothermic_mode:
            st.session_state.peak_y = ENDOTHERMIC_PEAK_ENERGY + (st.session_state.product_energy - ENDOTHERMIC_PRODUCT_ENERGY) * 1.1
        elif st.session_state.exothermic_mode:
            delta_E = st.session_state.reactant_energy - st.session_state.product_energy
            k = 0.15
            st.session_state.peak_y = st.session_state.reactant_energy + (EXOTHERMIC_PEAK_ENERGY - EXOTHERMIC_REACTANT_ENERGY) * np.exp(-k * delta_E)
        
        update_hammond_postulate()
        st.rerun()

with slider_col3:
    if not (st.session_state.endothermic_mode or st.session_state.exothermic_mode):
        min_peak = max(st.session_state.reactant_energy, st.session_state.product_energy) + 0.1
        new_peak = st.slider(
            "Transition State Energy",
            min_value=float(min_peak),
            max_value=18.0,
            value=float(st.session_state.peak_y),
            step=0.1,
            key="peak_slider"
        )
        if new_peak != st.session_state.peak_y:
            st.session_state.peak_y = new_peak
            st.rerun()
    else:
        st.write(f"TS Energy: {st.session_state.peak_y:.1f} (Auto-adjusted in mode)")

# Create and display the interactive plot
fig = create_interactive_plot()

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Control buttons below the graph
st.markdown("### Control Options")

# Control buttons in columns
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    ea_button_color = "🟢" if st.session_state.show_ea else "⚪"
    if st.button(f"{ea_button_color} Show/Hide Ea"):
        st.session_state.show_ea = not st.session_state.show_ea
        st.rerun()

with col2:
    deltaH_button_color = "🟠" if st.session_state.show_deltaH else "⚪"
    if st.button(f"{deltaH_button_color} Show/Hide ΔH"):
        st.session_state.show_deltaH = not st.session_state.show_deltaH
        st.rerun()

with col3:
    labels_button_color = "🟣" if st.session_state.show_labels else "⚪"
    if st.button(f"{labels_button_color} Show/Hide Labels"):
        st.session_state.show_labels = not st.session_state.show_labels
        st.rerun()

with col4:
    endo_button_color = "🔵" if st.session_state.endothermic_mode else "⚪"
    if st.button(f"{endo_button_color} Endothermic Mode"):
        st.session_state.endothermic_mode = not st.session_state.endothermic_mode
        if st.session_state.endothermic_mode:
            st.session_state.exothermic_mode = False
            st.session_state.reactant_energy = ENDOTHERMIC_REACTANT_ENERGY
            st.session_state.product_energy = ENDOTHERMIC_PRODUCT_ENERGY
            st.session_state.peak_y = ENDOTHERMIC_PEAK_ENERGY
        else:
            st.session_state.reactant_energy = 10
            st.session_state.product_energy = 10
            st.session_state.peak_y = 11.4
        update_hammond_postulate()
        st.rerun()

with col5:
    exo_button_color = "🔴" if st.session_state.exothermic_mode else "⚪"
    if st.button(f"{exo_button_color} Exothermic Mode"):
        st.session_state.exothermic_mode = not st.session_state.exothermic_mode
        if st.session_state.exothermic_mode:
            st.session_state.endothermic_mode = False
            st.session_state.reactant_energy = EXOTHERMIC_REACTANT_ENERGY
            st.session_state.product_energy = EXOTHERMIC_PRODUCT_ENERGY
            st.session_state.peak_y = EXOTHERMIC_PEAK_ENERGY
        else:
            st.session_state.reactant_energy = 10
            st.session_state.product_energy = 10
            st.session_state.peak_y = 11.4
        update_hammond_postulate()
        st.rerun()

# Additional control buttons
col6, col7, col8 = st.columns([1, 1, 2])

with col6:
    if st.button("💾 Save Curve"):
        x, y = compute_energy()
        deltaH_value = st.session_state.product_energy - st.session_state.reactant_energy
        curve_color = get_color_for_deltaH(deltaH_value)
        
        st.session_state.saved_curves.append({
            'x': x.tolist(),
            'y': y.tolist(),
            'color': curve_color
        })
        st.success(f"Curve saved! Total saved: {len(st.session_state.saved_curves)}")
        st.rerun()

with col7:
    if st.button("🗑️ Clear All Curves"):
        st.session_state.saved_curves = []
        st.success("All saved curves cleared!")
        st.rerun()

with col8:
    if len(st.session_state.saved_curves) > 0:
        st.info(f"📊 {len(st.session_state.saved_curves)} curve(s) saved")

# Display current values
st.markdown("### Current Energy Values")
info_col1, info_col2, info_col3, info_col4 = st.columns(4)

with info_col1:
    st.metric("Reactant Energy", f"{st.session_state.reactant_energy:.1f}")

with info_col2:
    st.metric("Product Energy", f"{st.session_state.product_energy:.1f}")

with info_col3:
    st.metric("Transition State Energy", f"{st.session_state.peak_y:.1f}")

with info_col4:
    delta_h = st.session_state.product_energy - st.session_state.reactant_energy
    st.metric("ΔH", f"{delta_h:.1f}", 
              delta=f"{'Endothermic' if delta_h > 0 else 'Exothermic' if delta_h < 0 else 'No change'}")

# Help section
with st.expander("ℹ️ Help & Features"):
    st.markdown("""
    **This interactive tool demonstrates:**
    - **Energy profiles** of chemical reactions
    - **Activation energy (Ea)** - energy barrier for the reaction
    - **Enthalpy change (ΔH)** - overall energy difference
    - **Hammond's postulate** - transition state position shifts with reaction favorability
    - **Endothermic vs Exothermic** reaction types
    
    **Controls:**
    - **Drag the colored circles** to adjust energies (when available)
    - **Toggle buttons** to show/hide different features
    - **Mode buttons** to switch between reaction types
    - **Save curves** to compare different reaction profiles
    
    **Colors:**
    - Blue: Reactant state
    - Red: Product state  
    - Purple/Dynamic: Transition state (color changes with position)
    - Green: Activation energy (Ea)
    - Orange: Enthalpy change (ΔH)
    """)