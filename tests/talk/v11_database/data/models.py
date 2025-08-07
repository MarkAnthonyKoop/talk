import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ValidationError(Exception):
    """Base class for validation errors."""
    pass

class DataType:
    """Represents a data type for a column."""

    def __init__(self, name: str, size: Optional[int] = None):
        self.name = name
        self.size = size

    def validate(self, value: Any) -> None:
        """Validates the given value against the data type.

        Args:
            value: The value to validate.

        Raises:
            ValidationError: If the value is invalid.
        """
        if self.name == "INTEGER":
            if not isinstance(value, int):
                raise ValidationError(f"Value '{value}' is not an integer.")
        elif self.name == "TEXT":
            if not isinstance(value, str):
                raise ValidationError(f"Value '{value}' is not a string.")
            if self.size is not None and len(value) > self.size:
                raise ValidationError(f"Text value '{value}' exceeds maximum length of {self.size}.")
        elif self.name == "REAL":
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Value '{value}' is not a real number.")
        # Add more data type validations as needed

    def __repr__(self):
        return f"DataType(name='{self.name}', size={self.size})"

class Column:
    """Represents a column in a table."""

    def __init__(self, name: str, data_type: DataType, primary_key: bool = False, nullable: bool = True):
        self.name = name
        self.data_type = data_type
        self.primary_key = primary_key
        self.nullable = nullable

    def __repr__(self):
        return f"Column(name='{self.name}', data_type={self.data_type}, primary_key={self.primary_key}, nullable={self.nullable})"

class Index:
    """Represents an index on a table."""

    def __init__(self, name: str, column_name: str, unique: bool = False):
        self.name = name
        self.column_name = column_name
        self.unique = unique

    def __repr__(self):
        return f"Index(name='{self.name}', column_name='{self.column_name}', unique={self.unique})"

class TableSchema:
    """Represents the schema of a table."""

    def __init__(self, name: str, columns: List[Column], indexes: List[Index] = []):
        self.name = name
        self.columns: Dict[str, Column] = {col.name: col for col in columns} #Dictionary for faster column lookup
        self.indexes = indexes

    def validate_row(self, row: Dict[str, Any]) -> None:
        """Validates a row of data against the table schema.

        Args:
            row: A dictionary representing a row of data.

        Raises:
            ValidationError: If the row is invalid.
        """
        for column_name, column in self.columns.items():
            value = row.get(column_name)
            if value is None:
                if not column.nullable:
                    raise ValidationError(f"Column '{column_name}' cannot be null.")
            else:
                try:
                    column.data_type.validate(value)
                except ValidationError as e:
                    raise ValidationError(f"Validation error for column '{column_name}': {e}")

    def __repr__(self):
        return f"TableSchema(name='{self.name}', columns={list(self.columns.values())}, indexes={self.indexes})"