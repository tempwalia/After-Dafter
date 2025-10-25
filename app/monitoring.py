import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

def setup_logging(app):
    """Configure logging for the application."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set up logging
    log_file = log_dir / f"app.log"
    
    # Configure handler
    handler = RotatingFileHandler(
        log_file,
        maxBytes=10000000,  # 10MB
        backupCount=5
    )
    
    # Set format
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Set up app logger
    app.logger.addHandler(handler)
    
    if app.debug:
        app.logger.setLevel(logging.DEBUG)
    else:
        app.logger.setLevel(logging.INFO)
        
    return app

def log_ml_activity(app, model_name, action, status, details=None):
    """Log ML model related activities."""
    
    message = f"ML Activity - Model: {model_name}, Action: {action}, Status: {status}"
    if details:
        message += f", Details: {details}"
    
    if status == "success":
        app.logger.info(message)
    elif status == "error":
        app.logger.error(message)
    else:
        app.logger.warning(message)