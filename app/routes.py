# app/routes.py
import os
import uuid
import datetime
import glob
from flask import Blueprint, current_app, render_template, request, redirect, url_for, send_from_directory, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import subprocess
import sys
import shlex

main_bp = Blueprint('main', __name__)

# Default dataset path (user uploaded file can override)
DEFAULT_DATA = os.path.join(os.getcwd(), 'data', 'synthetic_callcenter_accounts.csv')

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = current_app.config['USERS'].get(username)
        
        if user and user.password == password:
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('main.index'))
        else:
            flash('Invalid username or password.', 'danger')
    
    # Show login hints for demo purposes
    users = current_app.config['USERS']
    hints = [{'username': u.username, 'password': u.password} for u in users.values()]
    return render_template('login.html', hints=hints)

@main_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.login'))

@main_bp.route('/')
@login_required
def index():
    # Get list of available report HTML files (simple filename list)
    reports_path = os.path.join(current_app.config['ML_MODELS_DIR'], '*.html')
    model_files = sorted([os.path.basename(p) for p in glob.glob(reports_path)], reverse=True)
    
    # Get list of executable scripts with last run time
    # Use the actual scripts directory in the repo
    scripts_path = os.path.join(os.getcwd(), 'scripts', '*.py')
    scripts = []
    log_file = os.path.join(current_app.config['LOGS_DIR'], f'script_logs_{datetime.datetime.now().strftime("%Y%m%d")}.txt')
    
    # Read execution logs for last run times
    script_last_run = {}
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if 'executed' in line:
                    parts = line.split()
                    timestamp = ' '.join(parts[0:2]).strip('[]')
                    script_name = parts[parts.index('executed') + 1]
                    script_last_run[script_name] = timestamp
    
    for file in glob.glob(scripts_path):
        filename = os.path.basename(file)
        scripts.append({
            'name': filename,
            'path': file,
            'last_run': script_last_run.get(filename)
        })

    # List available notebook templates (for admin to run)
    notebooks_dir = os.path.join(os.getcwd(), 'notebooks')
    notebook_templates = []
    if os.path.isdir(notebooks_dir):
        for nb in sorted(os.listdir(notebooks_dir)):
            if nb.endswith('.ipynb'):
                notebook_templates.append({'name': nb, 'path': os.path.join(notebooks_dir, nb)})
    
    # Get recent execution logs
    execution_logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if 'executed' in line:
                    try:
                        parts = line.split()
                        timestamp = ' '.join(parts[0:2]).strip('[]')
                        user = parts[parts.index('User') + 1]
                        script = parts[parts.index('executed') + 1]
                        # Check if status exists in the log entry
                        status = 'Completed' if 'Status:' not in line else ' '.join(parts[parts.index('Status:') + 1:]).strip()
                        execution_logs.append({
                            'timestamp': timestamp,
                            'user': user,
                            'script_name': script,
                            'status': status
                        })
                    except (ValueError, IndexError):
                        # Skip malformed log entries
                        continue
    execution_logs = sorted(execution_logs, key=lambda x: x['timestamp'], reverse=True)[:5]  # Show last 5 executions
    
    return render_template(
        'index.html',
        model_files=model_files,
        scripts=scripts,
        execution_logs=execution_logs,
        notebooks=notebook_templates,
        default_data=DEFAULT_DATA,
    )

@main_bp.route('/execute-scripts', methods=['POST'])
@login_required
def execute_selected_scripts():
    selected_scripts = request.form.getlist('selected_scripts')
    if not selected_scripts:
        flash('No scripts selected.', 'warning')
        return redirect(url_for('main.index'))
    
    results = []
    for script_name in selected_scripts:
        # Execute scripts from the actual 'scripts' directory
        script_path = os.path.join(os.getcwd(), 'scripts', script_name)
        if not os.path.exists(script_path):
            results.append({'script': script_name, 'status': 'error', 'message': 'Script not found'})
            continue
        
        # Log the execution
        log_file = os.path.join(current_app.config['LOGS_DIR'], f'script_logs_{datetime.datetime.now().strftime("%Y%m%d")}.txt')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f'[{timestamp}] User {current_user.username} executed {script_name}'
        
        try:
            subprocess.run([sys.executable, script_path], check=True)
            status = 'success'
            message = 'Successfully executed'
        except subprocess.CalledProcessError as e:
            status = 'error'
            message = f'Error: {str(e)}'
        
        results.append({'script': script_name, 'status': status, 'message': message})
        
        # Update log file
        with open(log_file, 'a') as f:
            f.write(f'{log_entry} - Status: {status}\n')
    
    # Flash messages for all results
    for result in results:
        category = 'success' if result['status'] == 'success' else 'danger'
        flash(f"{result['script']}: {result['message']}", category)
    
    return redirect(url_for('main.index'))

@main_bp.route('/view-report/<filename>')
@login_required
def view_report(filename):
    # Get list of all available reports
    reports_path = os.path.join(current_app.config['ML_MODELS_DIR'], '*.html')
    all_reports = []
    current_report = None
    
    for file in glob.glob(reports_path):
        report_filename = os.path.basename(file)
        report_type = 'KMeans Clustering' if 'kmeans' in report_filename.lower() else 'XGBoost Prediction'
        description = f"Interactive {report_type} report with visualizations and metrics"
        report_info = {
            'name': report_filename,
            'date': datetime.datetime.fromtimestamp(os.path.getctime(file)).strftime('%Y-%m-%d %H:%M:%S'),
            'description': description
        }
        all_reports.append(report_info)
        if report_filename == filename:
            current_report = report_info
    
    if not current_report:
        flash('Report not found.', 'danger')
        return redirect(url_for('main.index'))
    
    # Create a URL for the report content
    report_content_url = url_for('main.get_report_content', filename=filename)
    
    return render_template('view_report.html', 
                         all_reports=sorted(all_reports, key=lambda x: x['date'], reverse=True),
                         current_report=current_report,
                         report_content_url=report_content_url)

@main_bp.route('/report-content/<filename>')
@login_required
def get_report_content(filename):
    """Serve the actual report content"""
    return send_from_directory(current_app.config['ML_MODELS_DIR'], filename)

@main_bp.route('/upload', methods=['GET','POST'])
@login_required
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            dest = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(dest)
            flash(f'Uploaded {filename}', 'success')
            return redirect(url_for('main.index'))
        else:
            flash('Please upload a CSV file.', 'danger')
    return render_template('upload.html')

@main_bp.route('/generate/kmeans', methods=['POST'])
def generate_kmeans():
    data_path = request.form.get('data_path') or DEFAULT_DATA
    # generate a unique name for the output HTML
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    output_html = f"kmeans_{timestamp}_{unique_id}.html"
    outpath = os.path.join(current_app.config['ML_MODELS_DIR'], output_html)

    # call the script synchronously - it will write the HTML report into outpath
    script = os.path.join(os.getcwd(), 'scripts', 'train_kmeans.py')
    cmd = [sys.executable, script, '--data', data_path, '--out', outpath]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        flash(f"KMeans generation failed: {proc.stderr}", "danger")
        return render_template('generating.html', name=output_html, status="failed", log=proc.stderr)
    flash(f"KMeans report generated: {output_html}", "success")
    return redirect(url_for('main.ml_models'))

@main_bp.route('/generate/xgb', methods=['POST'])
def generate_xgb():
    data_path = request.form.get('data_path') or DEFAULT_DATA
    # The XGBoost script expects clustering results (we will pass data path; script will read cluster col if present)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    output_html = f"xgb_{timestamp}_{unique_id}.html"
    outpath = os.path.join(current_app.config['ML_MODELS_DIR'], output_html)

    script = os.path.join(os.getcwd(), 'scripts', 'train_xgb.py')
    cmd = [sys.executable, script, '--data', data_path, '--out', outpath]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        flash(f"XGBoost generation failed: {proc.stderr}", "danger")
        return render_template('generating.html', name=output_html, status="failed", log=proc.stderr)
    flash(f"XGBoost report generated: {output_html}", "success")
    return redirect(url_for('main.ml_models'))

@main_bp.route('/ml-models')
def ml_models():
    models_dir = current_app.config['ML_MODELS_DIR']
    files = sorted([f for f in os.listdir(models_dir) if f.endswith('.html')], reverse=True)
    return render_template('ml_models.html', files=files)

@main_bp.route('/ml_models/<path:filename>')
def serve_model_report(filename):
    return send_from_directory(current_app.config['ML_MODELS_DIR'], filename)

import subprocess, datetime, uuid, shlex

def execute_notebook_and_export(template_path, params):
    # produce an executed notebook filename and an HTML output filename
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:6]
    exec_nb = os.path.join(current_app.config['ML_MODELS_DIR'], f"{os.path.splitext(os.path.basename(template_path))[0]}_exec_{ts}_{unique}.ipynb")
    out_html = os.path.join(current_app.config['ML_MODELS_DIR'], f"{os.path.splitext(os.path.basename(template_path))[0]}_{ts}_{unique}.html")

    # Build papermill command
    cmd_pm = [sys.executable, "-m", "papermill", template_path, exec_nb]
    # append params as -p key value
    for k, v in params.items():
        cmd_pm += ["-p", k, str(v)]

    # Run papermill to execute with parameters
    subprocess.run(cmd_pm, check=True)

    # Convert executed notebook to HTML
    cmd_nbconv = [sys.executable, "-m", "nbconvert", "--to", "html", exec_nb, "--output", os.path.basename(out_html)]
    # set output-dir so file lands in ML_MODELS_DIR
    subprocess.run(cmd_nbconv, check=True, cwd=current_app.config['ML_MODELS_DIR'])

    return os.path.basename(out_html)


@main_bp.route('/generate/notebook', methods=['POST'])
@login_required
def generate_notebook():
    # Only admin may execute notebooks
    if getattr(current_user, 'role', None) != 'admin':
        flash('Only admin users can generate notebook reports.', 'danger')
        return redirect(url_for('main.index'))

    notebook_name = request.form.get('notebook_name')
    data_path = request.form.get('data_path') or DEFAULT_DATA
    if not notebook_name:
        flash('No notebook selected.', 'warning')
        return redirect(url_for('main.index'))

    notebooks_dir = os.path.join(os.getcwd(), 'notebooks')
    template_path = os.path.join(notebooks_dir, notebook_name)
    if not os.path.exists(template_path):
        flash('Notebook template not found.', 'danger')
        return redirect(url_for('main.index'))

    try:
        params = {'data_path': data_path}
        out_html = execute_notebook_and_export(template_path, params)
        flash(f'Notebook executed and exported: {out_html}', 'success')
        return redirect(url_for('main.view_report', filename=out_html))
    except subprocess.CalledProcessError as e:
        flash(f'Notebook execution failed: {e}', 'danger')
        return redirect(url_for('main.index'))
