import logging
from functools import wraps
from flask import request, jsonify
from auth.jwt import verify_token

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def token_required(f):
    """
    Decorator to protect routes with JWT authentication.
    """
    @wraps(f)  # Preserve original function metadata
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            logging.warning("Token is missing in request headers.")
            return jsonify({'message': 'Token is missing'}), 401

        try:
            token = auth_header.split(" ")[1]  # Assuming "Bearer <token>" format
        except IndexError:
            logging.warning("Malformed token format in request headers.")
            return jsonify({'message': 'Malformed token'}), 400

        payload = verify_token(token)
        if not payload:
            logging.warning("Invalid token provided.")
            return jsonify({'message': 'Invalid token'}), 401

        # Pass the payload or specific user information to the decorated function
        kwargs['user_data'] = payload  # Pass the entire payload
        # Alternatively, pass specific user attributes:
        # kwargs['user_id'] = payload.get('user_id')
        # kwargs['username'] = payload.get('username')

        return f(*args, **kwargs)

    return decorated_function