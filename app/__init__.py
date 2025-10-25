# app/__init__.py
import os
from flask import Flask
from flask_login import LoginManager, UserMixin
import secrets
import os
from flask import Flask

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "devkey")
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'data')
    app.config['ML_MODELS_DIR'] = os.path.join(os.getcwd(), 'ml_models')


class User(UserMixin):
    def __init__(self, id, username, password, role):
        self.id = id
        self.username = username
        self.password = password
        self.role = role

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config['SECRET_KEY'] = secrets.token_hex(16)
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'data')
    app.config['ML_MODELS_DIR'] = os.path.join(os.getcwd(), 'ml_models')
    app.config['LOGS_DIR'] = os.path.join(os.getcwd(), 'logs')
    
    # Create necessary directories
    for dir_path in [app.config['UPLOAD_FOLDER'], app.config['ML_MODELS_DIR'], 
                    app.config['LOGS_DIR'], os.path.join(app.static_folder, 'model_plots')]:
        os.makedirs(dir_path, exist_ok=True)

    # Ensure notebooks directory exists for notebook templates
    notebooks_dir = os.path.join(os.getcwd(), 'notebooks')
    os.makedirs(notebooks_dir, exist_ok=True)

    # Setup Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'

    # Create demo users (in production, use proper password hashing)
    users = {
        'demo': User('1', 'demo', 'demo123', 'viewer'),
        'admin': User('2', 'admin', 'admin123', 'admin')
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
