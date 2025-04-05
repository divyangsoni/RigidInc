import io
import base64
from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-GUI environments (like web servers)
import matplotlib.pyplot as plt
try:
    from pileanalysis import perform_pile_cap_analysis, internal_defaults
except ImportError as e:
    print(f"FATAL ERROR: Could not import from Pileanalysis.py. Ensure it's in the correct directory.")
    print(f"Details: {e}")
    exit() 

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}
    form_data_to_repopulate = internal_defaults # Start with defaults for GET

    if request.method == 'POST':
        # Collect all form data into a dictionary - analysis function handles parsing/defaults
        form_data_input = request.form.to_dict()
        form_data_to_repopulate = form_data_input # Use submitted data to repopulate

        # Call the analysis function - it handles everything!
        results = perform_pile_cap_analysis(form_data_input)

        # --- Handle Matplotlib Plot ---
        if results.get('status') == 'success' and results.get('figure'):
            fig = results['figure']
            try:
                img = io.BytesIO()
                fig.savefig(img, format='png', bbox_inches='tight') # Save plot to buffer
                plt.close(fig)  # Close the figure to free memory
                img.seek(0)
                # Encode plot to base64 string to embed in HTML
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                results['plot_url'] = plot_url # Add plot data to results dictionary
            except Exception as plot_err:
                print(f"Error generating plot image: {plot_err}")
                results['warnings'] = results.get('warnings', []) + [f"Plot generation failed: {plot_err}"]
                results['plot_url'] = None # Ensure it's None if saving failed
            finally:
                 # Ensure figure/axes are not passed directly to template
                if 'figure' in results: del results['figure']
                if 'axes' in results: del results['axes']

        else:
             # Ensure plot_url is not present if analysis failed or plot wasn't generated
             results['plot_url'] = None
             # Ensure figure/axes are not passed directly to template
             if 'figure' in results: del results['figure']
             if 'axes' in results: del results['axes']

    # Render the template, passing the analysis results and form data (for defaults/repopulation)
    # We pass internal_defaults specifically for initial load, and form_data_to_repopulate for subsequent loads
    return render_template('index.html',
                           results=results,
                           form_data=form_data_to_repopulate,
                           defaults=internal_defaults) # Pass defaults for initial form values

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True) # debug=True is helpful for development