import logging
from flask import Flask

from api.auth_routes import auth_bp
from api.user_routes import user_bp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(user_bp)

@app.route('/')
def hello_world():
    """
    A simple hello world route.
    """
    return 'Hello, World!'

if __name__ == '__main__':
    logging.info("Starting the Flask application...")
    app.run(debug=True)  # Use debug=False in production
    logging.info("Flask application stopped.")