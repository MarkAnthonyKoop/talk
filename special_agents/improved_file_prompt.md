# File Editing Instructions

When editing files, use the SEARCH/REPLACE block format. This format allows precise edits while preserving indentation and formatting.

## Format Rules

### For editing existing files:

```
path/to/file.py
<<<<<<< SEARCH
exact lines to search for
including indentation
=======
replacement lines
with proper indentation
>>>>>>> REPLACE
```

### For creating new files:

```
path/to/newfile.py
<<<<<<< SEARCH
=======
entire file content here
with proper indentation
>>>>>>> REPLACE
```

## Important Guidelines

1. **Exact Matching**: The SEARCH section must contain the EXACT lines from the file, including all whitespace and indentation.

2. **Multiple Edits**: For multiple changes in the same file, use separate SEARCH/REPLACE blocks:

```
file.py
<<<<<<< SEARCH
old code block 1
=======
new code block 1
>>>>>>> REPLACE

file.py
<<<<<<< SEARCH
old code block 2
=======
new code block 2
>>>>>>> REPLACE
```

3. **Preserve Indentation**: Always maintain the correct indentation level in both SEARCH and REPLACE sections.

4. **Complete Lines**: Include complete lines in your SEARCH blocks, not partial lines.

5. **Unique Matches**: Include enough context in SEARCH blocks to ensure a unique match.

## Examples

### Example 1: Adding a method to a class

```
myapp/models.py
<<<<<<< SEARCH
class User:
    def __init__(self, name):
        self.name = name
=======
class User:
    def __init__(self, name):
        self.name = name
    
    def get_greeting(self):
        return f"Hello, {self.name}!"
>>>>>>> REPLACE
```

### Example 2: Creating a new file

```
myapp/utils.py
<<<<<<< SEARCH
=======
def format_date(date_obj):
    """Format a date object as a string."""
    return date_obj.strftime("%Y-%m-%d")

def parse_date(date_str):
    """Parse a date string into a date object."""
    from datetime import datetime
    return datetime.strptime(date_str, "%Y-%m-%d")
>>>>>>> REPLACE
```

### Example 3: Multiple edits

```
config.py
<<<<<<< SEARCH
DEBUG = False
=======
DEBUG = True
>>>>>>> REPLACE

config.py
<<<<<<< SEARCH
LOG_LEVEL = "INFO"
=======
LOG_LEVEL = "DEBUG"
>>>>>>> REPLACE
```

Remember: The SEARCH block must match EXACTLY what's in the file, character for character, including all spaces and indentation.