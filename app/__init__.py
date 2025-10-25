"""Flask app factory and configuration."""
import os
import secrets
import subprocess
import sys
from flask import Flask
from flask_login import LoginManager, UserMixin


class User(UserMixin):
    def __init__(self, id, username, password, role):
        self.id = id
        self.username = username
        self.password = password
        self.role = role


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Config from environment with sane defaults
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(16))
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'data')
    app.config['ML_MODELS_DIR'] = os.path.join(os.getcwd(), 'ml_models')
    app.config['LOGS_DIR'] = os.path.join(os.getcwd(), 'logs')

    # Create necessary directories
    for dir_path in [
        app.config['UPLOAD_FOLDER'],
        app.config['ML_MODELS_DIR'],
        app.config['LOGS_DIR'],
        os.path.join(app.static_folder, 'model_plots'),
    ]:
        os.makedirs(dir_path, exist_ok=True)

    # Ensure notebooks directory exists for notebook templates
    notebooks_dir = os.path.join(os.getcwd(), 'notebooks')
    os.makedirs(notebooks_dir, exist_ok=True)

    # Seed sample data on first run (especially for Render) if missing
    sample_csv = os.path.join(app.config['UPLOAD_FOLDER'], 'synthetic_callcenter_accounts.csv')
    if not os.path.exists(sample_csv):
        try:
            setup_script = os.path.join(os.getcwd(), 'scripts', 'setup_sample_data.py')
            if os.path.exists(setup_script):
                subprocess.run([sys.executable, setup_script], check=True)
        except Exception as e:
            # Non-fatal: app can still run; logs will help diagnose
            try:
                os.makedirs(app.config['LOGS_DIR'], exist_ok=True)
                with open(os.path.join(app.config['LOGS_DIR'], 'startup.log'), 'a') as lf:
                    lf.write(f"Failed to seed sample data: {e}\n")
            except Exception:
                pass

    # Setup Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'

    # Users from environment (for demo purposes; use hashing in production)
    admin_user = os.getenv('ADMIN_USERNAME', 'admin')
    admin_pass = os.getenv('ADMIN_PASSWORD', 'admin123')
    demo_user = os.getenv('DEMO_USERNAME', 'demo')
    demo_pass = os.getenv('DEMO_PASSWORD', 'demo123')

    users = {
        'demo': User('1', demo_user, demo_pass, 'viewer'),
        'admin': User('2', admin_user, admin_pass, 'admin'),
    }

    @login_manager.user_loader
    def load_user(user_id):
        for user in users.values():
            if user.id == user_id:
                return user
        return None

    app.config['USERS'] = users

    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app
