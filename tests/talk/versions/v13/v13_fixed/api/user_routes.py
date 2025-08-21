import logging
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash
from auth.jwt import verify_token
from models.user import Database, User

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

user_bp = Blueprint('user', __name__, url_prefix='/users')

# Initialize database (replace with your actual database URL)
db = Database()

def token_required(f):
    """
    Decorator to protect routes with JWT authentication.
    """
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'message': 'Token is missing'}), 401

        try:
            token = auth_header.split(" ")[1]
            payload = verify_token(token)
            if not payload:
                return jsonify({'message': 'Invalid token'}), 401
        except IndexError:
            return jsonify({'message': 'Malformed token'}), 400

        return f(*args, **kwargs, user_id=payload.get('user_id'))  # Pass user_id to the route

    decorated_function.__name__ = f.__name__  # Preserve original function name
    return decorated_function


@user_bp.route('/<int:user_id>', methods=['GET'])
@token_required
def get_user(user_id: int, user_id_from_token: int):
    """
    Retrieves a user by ID.
    """
    if user_id != user_id_from_token:
        return jsonify({'message': 'Unauthorized'}), 403

    user = db.db.query(User).filter(User.id == user_id).first() # Directly access db session

    if user:
        return jsonify({'id': user.id, 'username': user.username, 'email': user.email}), 200
    else:
        return jsonify({'message': 'User not found'}), 404

@user_bp.route('/<int:user_id>', methods=['PUT'])
@token_required
def update_user(user_id: int, user_id_from_token: int):
    """
    Updates a user's information.
    """
    if user_id != user_id_from_token:
        return jsonify({'message': 'Unauthorized'}), 403

    user = db.db.query(User).filter(User.id == user_id).first()

    if not user:
        return jsonify({'message': 'User not found'}), 404

    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if username:
        user.username = username
    if email:
        user.email = email
    if password:
        user.password_hash = generate_password_hash(password)

    try:
        db.db.commit()
        return jsonify({'message': 'User updated successfully'}), 200
    except Exception as e:
        db.db.rollback()
        logging.error(f"Error updating user: {e}")
        return jsonify({'message': 'Failed to update user'}), 500

@user_bp.route('/<int:user_id>', methods=['DELETE'])
@token_required
def delete_user(user_id: int, user_id_from_token: int):
    """
    Deletes a user.
    """
    if user_id != user_id_from_token:
        return jsonify({'message': 'Unauthorized'}), 403

    user = db.db.query(User).filter(User.id == user_id).first()

    if not user:
        return jsonify({'message': 'User not found'}), 404

    try:
        db.db.delete(user)
        db.db.commit()
        return jsonify({'message': 'User deleted successfully'}), 200
    except Exception as e:
        db.db.rollback()
        logging.error(f"Error deleting user: {e}")
        return jsonify({'message': 'Failed to delete user'}), 500