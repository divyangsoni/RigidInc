import io
import base64
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from pileanalysis import perform_pile_cap_analysis
except ImportError as e:
    print(f"FATAL ERROR: Could not import from Pileanalysis.py. Ensure it's in the correct directory.")
    print(f"Details: {e}")
    # In a server environment, you might want to handle this more gracefully
    # For now, we'll let it raise an exception on startup.
    raise

app = Flask(__name__)

# --- ADD THIS LINE ---
# This will enable CORS for all routes and all origins.
CORS(app) 
# For more security in production, you might restrict the origins:
# CORS(app, resources={r"/api/*": {"origins": "https://your-react-app-domain.vercel.app"}})
# ---------------------


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    API endpoint to perform the pile cap analysis.
    Accepts form data as JSON and returns results as JSON.
    """
    try:
        # Get data from the JSON request body
        form_data_input = request.json
        if not form_data_input:
            return jsonify({'status': 'error', 'message': 'No input data provided in request body.'}), 400

        # The analysis function handles everything
        results = perform_pile_cap_analysis(form_data_input)

        # Handle the plot if it was generated successfully
        if results.get('status') == 'success' and results.get('figure'):
            fig = results['figure']
            try:
                img = io.BytesIO()
                fig.savefig(img, format='png', bbox_inches='tight')
                plt.close(fig)
                img.seek(0)
                # Encode plot to base64 string to embed in the response
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                results['plot_url'] = plot_url
            except Exception as plot_err:
                print(f"Error generating plot image: {plot_err}")
                results['warnings'] = results.get('warnings', []) + [f"Plot generation failed: {plot_err}"]
                results['plot_url'] = None
            finally:
                # Remove non-serializable objects before returning JSON
                if 'figure' in results: del results['figure']
                if 'axes' in results: del results['axes']
        else:
            if 'figure' in results: del results['figure']
            if 'axes' in results: del results['axes']

        return jsonify(results)

    except Exception as e:
        # General error handler for any other exceptions during analysis
        print(f"An error occurred during analysis: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'An internal server error occurred: {e}'}), 500

# The root endpoint can be a simple health check now
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Flask analysis server is running.'
    })

# This part is for local development, Vercel will handle running the app.
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

