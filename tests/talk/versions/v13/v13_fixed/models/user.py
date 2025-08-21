import logging
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base for declarative models
Base = declarative_base()

class User(Base):
    """
    Represents a user in the system.
    """
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)  # Store password hashes, not plain passwords
    email = Column(String(120), unique=True, nullable=False)

    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"

class Database:
    """
    Manages the database connection and operations.
    """
    def __init__(self, db_url: str = 'sqlite:///:memory:'):
        """
        Initializes the database connection.

        Args:
            db_url: The URL for the database. Defaults to an in-memory SQLite database.
        """
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        logging.info(f"Database connection established to {db_url}")

    def add_user(self, username: str, password_hash: str, email: str) -> Optional[User]:
        """
        Adds a new user to the database.

        Args:
            username: The username of the new user.
            password_hash: The password hash of the new user.
            email: The email address of the new user.

        Returns:
            The newly created User object if successful, None otherwise.
        """
        try:
            new_user = User(username=username, password_hash=password_hash, email=email)
            self.db.add(new_user)
            self.db.commit()
            logging.info(f"User '{username}' added successfully.")
            return new_user
        except Exception as e:
            self.db.rollback()
            logging.error(f"Error adding user '{username}': {e}")
            return None

    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Retrieves a user from the database by username.

        Args:
            username: The username of the user to retrieve.

        Returns:
            The User object if found, None otherwise.
        """
        try:
            user = self.db.query(User).filter(User.username == username).first()
            if user:
                logging.info(f"User '{username}' retrieved successfully.")
            else:
                logging.info(f"User '{username}' not found.")
            return user
        except Exception as e:
            logging.error(f"Error retrieving user '{username}': {e}")
            return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Retrieves a user from the database by email.

        Args:
            email: The email of the user to retrieve.

        Returns:
            The User object if found, None otherwise.
        """
        try:
            user = self.db.query(User).filter(User.email == email).first()
            if user:
                logging.info(f"User with email '{email}' retrieved successfully.")
            else:
                logging.info(f"User with email '{email}' not found.")
            return user
        except Exception as e:
            logging.error(f"Error retrieving user with email '{email}': {e}")
            return None

    def close(self):
        """
        Closes the database session.
        """
        self.db.close()
        logging.info("Database connection closed.")