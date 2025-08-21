import jwt
import time
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace with a strong, randomly generated secret key in a production environment.
SECRET_KEY = "your-secret-key"  # NEVER store secrets directly in code!
ALGORITHM = "HS256"
TOKEN_EXPIRATION_TIME = 3600  # Token expires in 1 hour (seconds)

def generate_token(payload: Dict, expiry: int = TOKEN_EXPIRATION_TIME) -> Optional[str]:
    """
    Generates a JWT token.

    Args:
        payload: The payload to include in the token (e.g., user ID, username).  Must be JSON serializable.
        expiry: The expiration time for the token, in seconds. Defaults to TOKEN_EXPIRATION_TIME.

    Returns:
        The generated JWT token as a string, or None if an error occurs.
    """
    try:
        payload_with_exp = payload.copy()  # Avoid modifying the original payload
        payload_with_exp['exp'] = time.time() + expiry
        token = jwt.encode(payload_with_exp, SECRET_KEY, algorithm=ALGORITHM)
        logging.info(f"Token generated successfully for payload: {payload}")
        return token
    except Exception as e:
        logging.error(f"Error generating token: {e}")
        return None


def verify_token(token: str) -> Optional[Dict]:
    """
    Verifies a JWT token and returns the decoded payload.

    Args:
        token: The JWT token to verify.

    Returns:
        The decoded payload as a dictionary if the token is valid, or None if the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        logging.info("Token verified successfully.")
        return payload
    except jwt.ExpiredSignatureError:
        logging.warning("Token has expired.")
        return None
    except jwt.InvalidTokenError:
        logging.warning("Invalid token.")
        return None
    except Exception as e:
        logging.error(f"Error verifying token: {e}")
        return None


if __name__ == '__main__':
    # Example usage
    user_data = {"user_id": 123, "username": "testuser"}
    token = generate_token(user_data)

    if token:
        print("Generated Token:", token)

        # Simulate waiting for the token to expire (optional)
        # time.sleep(TOKEN_EXPIRATION_TIME + 1)

        decoded_payload = verify_token(token)

        if decoded_payload:
            print("Decoded Payload:", decoded_payload)
        else:
            print("Token verification failed.")
    else:
        print("Token generation failed.")