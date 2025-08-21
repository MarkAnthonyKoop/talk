import logging
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

from models.user import Database
from auth.jwt import generate_token, verify_token

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# Initialize database (replace with your actual database URL)
db = Database()

@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Registers a new user.
    """
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')

    if not username or not password or not email:
        return jsonify({'message': 'Missing username, password, or email'}), 400

    if db.get_user_by_username(username):
        return jsonify({'message': 'Username already exists'}), 400

    if db.get_user_by_email(email):
        return jsonify({'message': 'Email already exists'}), 400

    password_hash = generate_password_hash(password)
    new_user = db.add_user(username, password_hash, email)

    if new_user:
        return jsonify({'message': 'User registered successfully'}), 201
    else:
        return jsonify({'message': 'Failed to register user'}), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Logs in an existing user.
    """
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Missing username or password'}), 400

    user = db.get_user_by_username(username)

    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'message': 'Invalid username or password'}), 401

    # Generate JWT token
    token_payload = {'user_id': user.id, 'username': user.username}
    token = generate_token(token_payload)

    if token:
        return jsonify({'message': 'Login successful', 'token': token}), 200
    else:
        return jsonify({'message': 'Failed to generate token'}), 500

@auth_bp.route('/logout', methods=['POST'])
def logout():
    """
    Logs out the user.  In a stateless JWT setup, "logging out" usually means
    the client discarding the token.  This endpoint is mostly for client-side
    cleanup or invalidating the token on the server (if you implement a blacklist).
    """
    # In a simple JWT setup, there's nothing to do on the server-side to "logout".
    # The client simply discards the token.
    # If you implement a token blacklist, you would add the token to the blacklist here.

    # Get token from Authorization header
    auth_header = request.headers.get('Authorization')
    if auth_header:
        try:
            token = auth_header.split(" ")[1]  # Assuming "Bearer <token>" format
        except IndexError:
            return jsonify({'message': 'Malformed token'}), 400
    else:
        return jsonify({'message': 'Token required'}), 400

    # In a real implementation, you might add the token to a blacklist here.
    # For this example, we just return a success message.
    logging.info(f"Logout requested.  Token: {token}") #In a real implementation, you would need to invalidate the token.

    return jsonify({'message': 'Logout successful'}), 200