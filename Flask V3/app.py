import io
import base64
from flask import Flask, render_template, request, redirect, url_for, session, flash
from functools import wraps # For login_required decorator
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
# Secret key is needed for session management. Change this to a random string!
app.secret_key = 'your_very_secret_random_string_here_change_me'

# --- Hardcoded Credentials ---
VALID_USERNAME = "Snell"
VALID_PASSWORD = "1517State!"

# --- Login Required Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# --- Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            flash('You were successfully logged in!', 'success')
            next_url = request.args.get('next')
            return redirect(next_url or url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required # Protect this route
def index():
    results = {}
    form_data_to_repopulate = internal_defaults # Start with defaults for GET

    if request.method == 'POST':
        form_data_input = request.form.to_dict()
        form_data_to_repopulate = form_data_input

        results = perform_pile_cap_analysis(form_data_input)

        if results.get('status') == 'success' and results.get('figure'):
            fig = results['figure']
            try:
                img = io.BytesIO()
                fig.savefig(img, format='png', bbox_inches='tight')
                plt.close(fig)
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                results['plot_url'] = plot_url
            except Exception as plot_err:
                print(f"Error generating plot image: {plot_err}")
                results['warnings'] = results.get('warnings', []) + [f"Plot generation failed: {plot_err}"]
                results['plot_url'] = None
            finally:
                if 'figure' in results: del results['figure']
                if 'axes' in results: del results['axes']
        else:
            results['plot_url'] = None
            if 'figure' in results: del results['figure']
            if 'axes' in results: del results['axes']

    return render_template('index.html',
                           results=results,
                           form_data=form_data_to_repopulate,
                           defaults=internal_defaults)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)