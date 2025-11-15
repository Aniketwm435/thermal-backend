# Set the backend for Matplotlib to 'Agg'
# This is CRITICAL for running in a headless server environment (like Render.com)
# It must be done BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import io
from flask import Flask, request, send_file, jsonify

# Initialize the Flask app
app = Flask(_name_)

def create_earth_depth_plot():
    """
    Contains the exact plotting logic you provided.
    Generates the plot and returns the Matplotlib figure object.
    """
    
    # --- 1. CONFIGURATION ---
    np.random.seed(678) 
    depth_min, depth_max = 0, 80
    surface_x_min, surface_x_max = 1, 6
    n_points = 800  
    grid_resolution = 80 
    cbar_ticks = [55, 71, 93, 121, 157, 205, 267, 348, 453, 590, 768, 1000]

    # --- 2. DATA GENERATION ---
    depth_grid = np.linspace(depth_min, depth_max, grid_resolution)
    surface_x_grid = np.linspace(surface_x_min, surface_x_max, grid_resolution)
    X_grid, Z_grid = np.meshgrid(surface_x_grid, depth_grid)
    points_x = np.random.uniform(surface_x_min, surface_x_max, n_points)
    points_z = np.random.uniform(depth_min, depth_max, n_points)
    points = np.vstack((points_x, points_z)).T
    values = np.zeros(n_points)

    boundary_z_upper = (40 + 15 * np.sin(points_x * np.pi / 2.5) + 10 * np.random.randn(n_points) * 0.5)
    boundary_z_upper = np.clip(boundary_z_upper, 25, 50)
    is_hard_rock_zone = points_z < boundary_z_upper
    values[is_hard_rock_zone] = 800 + 350 * np.random.rand(np.sum(is_hard_rock_zone))
    is_deep_zone = points_z >= boundary_z_upper
    values[is_deep_zone] = 450 + 250 * np.random.rand(np.sum(is_deep_zone)) 
    is_pocket_1 = is_deep_zone & (points_x > 1.5) & (points_x < 3.5) & (points_z > 50) & (points_z < 70)
    is_pocket_1 &= (np.random.rand(n_points) > 0.4) 
    values[is_pocket_1] = 50 + 70 * np.random.rand(np.sum(is_pocket_1))
    is_pocket_2 = is_deep_zone & (points_x > 4.5) & (points_x < 6.0) & (points_z > 55) & (points_z < 75)
    is_pocket_2 &= (np.random.rand(n_points) > 0.4) 
    values[is_pocket_2] = 100 + 100 * np.random.rand(np.sum(is_pocket_2))
    is_deepest_layer = points_z >= 75
    values[is_deepest_layer] = 600 + 300 * np.random.rand(np.sum(is_deepest_layer))
    values += 100 * np.random.randn(n_points) 

    data_interpolated = griddata(points, values, (X_grid, Z_grid), method='linear')
    data_interpolated = np.clip(data_interpolated, 55, 1000)

    # --- 3. PLOTTING SETUP ---
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.1)
    ax_map = fig.add_subplot(gs[0, 0])

    levels = np.linspace(55, 1000, 15) 
    c = ax_map.contourf(X_grid, Z_grid, data_interpolated, levels=levels, cmap='jet')

    # --- Main Plot Configuration ---
    ax_map.set_xlabel('Surface-X (m/f)', fontsize=12)
    ax_map.set_ylabel('Depth-Z (m/f)', fontsize=12)
    ax_map.invert_yaxis() 
    ax_map.set_yticks(np.arange(0, 90, 10)) 
    ax_map.set_xticks(np.arange(1, 7, 1))
    ax_map.set_title('Earth Depth Profile', fontsize=16, weight='bold', pad=15)

    # --- 4. ADDING TEXT LABELS ---
    ax_map.text(2.0, -5, 'Soft Rock\nAnd Dry Sand', fontsize=10, color='black', ha='center', va='center')
    ax_map.text(5.0, -5, 'Wet Nature', fontsize=10, color='black', ha='center', va='center')
    ax_map.text(0.7, 20, 'More Hard', fontsize=10, color='black', rotation=90, va='center', ha='center')
    ax_map.text(2.0, 85, 'Most Hard\nStructure', fontsize=10, color='black', ha='center', va='center')
    ax_map.text(3.5, 85, 'Wet Condition', fontsize=10, color='black', ha='center', va='center')
    ax_map.text(5.0, 85, 'Water Bearing Rock', fontsize=10, color='black', ha='center', va='center')

    # --- 5. DETAILED COLORBAR LEGEND ---
    ax_legend = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(c, ax_legend, ticks=cbar_ticks, fraction=1.0, pad=0.0)
    cbar.set_label('Value', visible=False) 
    ax_legend.set_yticklabels([]) 
    ax_legend.set_xticks([]) 
    ax_legend.set_title('Legend', fontsize=12, pad=10)

    legend_text_map = {
        'Hard Rock': 900,
        'Medium Hard Rock': 750,
        'Less Medium Rock, Below Soft Rock': 650,
        'Rock, Soil and Wet Nature': 350,
        'Less Dense Porous Rock': 190,
        'Little More Dense Porous Rock': 130,
        'More Dense Porous Rock (Water Bearing Rock Layer)': 85
    }

    for text, y_pos in legend_text_map.items():
        ax_legend.text(1.2, y_pos, text, transform=ax_legend.transData,
                       fontsize=9, ha='left', va='center')
        ax_legend.hlines(y_pos, 0.95, 1.15, colors='black', lw=1, transform=ax_legend.get_yaxis_transform())
        ax_legend.hlines(y_pos, 0.1, 0.2, colors='black', lw=2, transform=ax_legend.get_yaxis_transform())

    # --- Descriptive Text Below Plot ---
    fig.text(0.5, 0.05, 
             'The Earth Depth Profile describes the spread of Soft rock, Hard rock and the Water Bearing Porous rock information.',
             fontsize=12, color='black', ha='center', va='top', wrap=True, transform=fig.transFigure)

    plt.tight_layout(rect=[0.05, 0.1, 1.0, 1.0])
    
    # Return the figure object
    return fig

# --- API Endpoint Definition ---

@app.route("/generate-pdf", methods=["POST"])
def generate_pdf():
    """
    API endpoint to generate the PDF and send it to the client.
    """
    try:
        # Get JSON data from the request (even if not used by this plot)
        # This matches the API contract for future use
        data = request.json
        # You could use data here if needed, e.g.,
        # location = data.get('location', 'Unknown')
    except Exception as e:
        return jsonify({"error": "Invalid JSON data", "message": str(e)}), 400

    try:
        # 1. Create the plot
        fig = create_earth_depth_plot()

        # 2. Save the plot to a in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='pdf')
        
        # 3. Rewind the buffer's "cursor" to the beginning
        buf.seek(0)
        
        # 4. IMPORTANT: Close the figure to free up memory
        plt.close(fig)

        # 5. Send the buffer as a file
        return send_file(
            buf,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='earth_depth_profile.pdf'
        )
    
    except Exception as e:
        # Close the plot in case of an error
        plt.close('all')
        return jsonify({"error": "Failed to generate plot", "message": str(e)}), 500

# Health check route
@app.route("/", methods=["GET"])
def health_check():
    return "Python PDF Generator is running."

# Main entry point for running the app (for local testing)
# Render.com will use gunicorn to run 'app:app'
if _name_ == "_main_":
    app.run(host='0.0.0.0', port=5000)
