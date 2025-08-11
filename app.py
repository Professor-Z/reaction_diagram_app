import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Define your functions from your script
def cubic_with_conditions(x0, y0, x1, y1, x2, y2, slope0=0, slope2=0):
    A = np.array([
        [x0**3, x0**2, x0, 1],
        [3*x0**2, 2*x0, 1, 0],
        [x2**3, x2**2, x2, 1],
        [3*x2**2, 2*x2, 1, 0]
    ])
    b = np.array([y0, slope0, y2, slope2])
    coeffs = np.linalg.solve(A, b)
    return np.poly1d(coeffs)

def generate_left_segment(reactant_energy, peak_y, peak_x, x):
    x_inflect = peak_x/2
    y_inflect = (reactant_energy + peak_y)/2
    return cubic_with_conditions(0, reactant_energy, x_inflect, y_inflect, peak_x, peak_y)

def generate_right_segment(p_energy, peak_y, peak_x, x):
    x_inflect = peak_x + (x[-1] - peak_x)/2
    y_inflect = (peak_y + p_energy)/2
    return cubic_with_conditions(peak_x, peak_y, x_inflect, y_inflect, x[-1], p_energy)

def get_transition_color(peak_x):
    t = (peak_x - 1) / (9 - 1)
    t = np.clip(t, 0, 1)
    if t <= 0.5:
        t2 = t / 0.5
        r = 0 + t2*0.5
        g = 0
        b = 1 + t2*(-0.5)
    else:
        t2 = (t - 0.5)/0.5
        r = 0.5 + t2*0.5
        g=0
        b=0.5 + t2*(-0.5)
    return r,g,b

# Main app function
def main():
    st.title("Reaction Coordinate Diagram")

    # Sliders for parameters
    reactant_energy = st.slider("Reactant Energy", 0.0, 10.0, 0.0, step=0.1)
    product_energy = st.slider("Product Energy", 0.0, 10.0, 5.0, step=0.1)
    peak_x = st.slider("Transition State X", 1.0, 9.0, 5.0, step=0.1)

    x = np.linspace(0, 10, 500)
    # Generate energy profile based on current slider values
    left_poly = generate_left_segment(reactant_energy, reactant_energy+1, peak_x, x)
    right_poly = generate_right_segment(product_energy, reactant_energy+1, peak_x, x)

    y = np.empty_like(x)
    mask_left = x <= peak_x
    mask_right = x > peak_x
    y[mask_left] = left_poly(x[mask_left])
    y[mask_right] = right_poly(x[mask_right])

    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x, y, lw=2)
    ax.set_xlim(-1,11)
    ax.set_ylim(-1, max(reactant_energy, product_energy, reactant_energy+1)+2)
    ax.set_xlabel('Reaction Coordinate', fontweight='bold')
    ax.set_ylabel('Energy', fontweight='bold')
    ax.set_title('Reaction Coordinate Diagram', fontweight='bold')

    # Plot points
    reactant_point, = ax.plot(0, reactant_energy, 'o', color='blue', markersize=12)
    product_point, = ax.plot(10, product_energy, 'o', color='red', markersize=12)
    peak_point, = ax.plot(peak_x, reactant_energy+1, 'o', markersize=12)

    # Labels
    reactant_label = ax.text(0, reactant_energy + 0.5, 'REACTANT', color='blue', weight='bold', ha='center')
    product_label = ax.text(10, product_energy - 0.5, 'PRODUCT', color='red', weight='bold', ha='center')
    ts_label = ax.text(peak_x, reactant_energy + 1 + 0.5, 'TS', color='purple', weight='bold', ha='center')

    # Draw initial Ea arrow
    x_arrow = peak_x - 0.5
    y_start = reactant_energy
    y_end = reactant_energy + 1
    ea_obj = ax.annotate(
        '', xy=(x_arrow, y_end), xytext=(x_arrow, y_start),
        arrowprops=dict(arrowstyle='<->', color='green', lw=2)
)
    # Add Ea label
    if 'ea_text' in globals():
        ea_text.remove()
    ea_text = ax.text(x_arrow - 0.1, (y_start + y_end)/2, r'$E_a$', color='green', weight='bold', ha='right', va='center')

    # Draw initial ΔH arrow
    x_deltaH = peak_x + 0.5
    y_start_deltaH = reactant_energy
    y_end_deltaH = product_energy
    deltaH_obj = ax.annotate(
        '', xy=(x_deltaH, y_end_deltaH), xytext=(x_deltaH, y_start_deltaH),
        arrowprops=dict(arrowstyle='<->', color='orange', lw=2)
)
    # Add ΔH label
    if 'deltaH_text' in globals():
        deltaH_text.remove()
    deltaH_text = ax.text(x_deltaH + 0.1, (y_start_deltaH + y_end_deltaH)/2, r'$\Delta H$', color='orange', weight='bold', ha='left', va='center')

    # Create a figure for the plot
    import streamlit as st

    def plot_reaction_diagram(reactant_E, product_E, ts_x):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8,6))
        # Generate the energy profile based on current sliders/parameters
        left_poly = generate_left_segment(reactant_E, reactant_E + 1, ts_x, x)
        right_poly = generate_right_segment(product_E, reactant_E + 1, ts_x, x)
        y = np.empty_like(x)
        mask_left = x <= ts_x
        mask_right = x > ts_x
        y[mask_left] = left_poly(x[mask_left])
        y[mask_right] = right_poly(x[mask_right])
        # Plot the profile
        ax.plot(x, y, lw=2)
        ax.set_xlim(-1,11)
        ax.set_ylim(-1, max(reactant_E, product_E, reactant_E+1)+2)
        ax.set_xlabel('Reaction Coordinate', fontweight='bold')
        ax.set_ylabel('Energy', fontweight='bold')
        ax.set_title('Reaction Coordinate Diagram', fontweight='bold')
        # Points
        react_point, = ax.plot(0, reactant_E, 'o', color='blue', markersize=12)
        prod_point, = ax.plot(10, product_E, 'o', color='red', markersize=12)
        peak_point, = ax.plot(ts_x, reactant_E + 1, 'o', markersize=12)
        # Labels
        ax.text(0, reactant_E+0.5, 'REACTANT', color='blue', weight='bold', ha='center')
        ax.text(10, product_E-0.5, 'PRODUCT', color='red', weight='bold', ha='center')
        ax.text(ts_x, reactant_E+1+0.5, 'TS', color='purple', weight='bold', ha='center')
        # Labels positions
        reactant_label.set_x(0)
        reactant_label.set_y(reactant_E + 0.5)
        product_label.set_x(10)
        product_label.set_y(product_E - 0.5)
        ts_label.set_x(ts_x)
        ts_label.set_y(reactant_E + 1 + 0.5)
        # Transition state marker color
        r, g, b = get_transition_color(ts_x)
        peak_point.set_markerfacecolor((r, g, b))
        peak_point.set_markeredgecolor((r, g, b))
    # Ea arrow
    if 'ea_obj' in globals():
        ea_obj.remove()
    ea_obj = ax.annotate(
        '', xy=(x_arrow, y_end), xytext=(x_arrow, y_start),
        arrowprops=dict(arrowstyle='<->', color='green', lw=2)
)
    # Ea label
    if 'ea_text' in globals():
        ea_text.remove()
    ea_text = ax.text(x_arrow - 0.1, (y_start + y_end) / 2, r'$E_a$', color='green', weight='bold', ha='right', va='center')

    # ΔH arrow
    if 'deltaH_obj' in globals():
        deltaH_obj.remove()
    deltaH_obj = ax.annotate(
        '', xy=(x_deltaH, y_end_deltaH), xytext=(x_deltaH, y_start_deltaH),
        arrowprops=dict(arrowstyle='<->', color='orange', lw=2)
)
    # ΔH label
    if 'deltaH_text' in globals():
        deltaH_text.remove()
    deltaH_text = ax.text(x_deltaH + 0.1, (y_start_deltaH + y_end_deltaH)/2, r'$\Delta H$', color='orange', weight='bold', ha='left', va='center')

    # Adjust label positions
    reactant_label.set_x(0)
    reactant_label.set_y(reactant_E + 0.5)

    product_label.set_x(10)
    product_label.set_y(product_E - 0.5)

    ts_label.set_x(ts_x)
    ts_label.set_y(reactant_E + 1 + 0.5)

    # Transition state marker color
    r, g, b = get_transition_color(ts_x)
    peak_point.set_markerfacecolor((r, g, b))
    peak_point.set_markeredgecolor((r, g, b))

    plt.draw()
