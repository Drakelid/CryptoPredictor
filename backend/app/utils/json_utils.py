import json
from datetime import datetime, date
import numpy as np

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that can handle:
    - datetime objects
    - date objects
    - numpy arrays and scalars
    - infinity values
    """
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif obj == float('inf') or obj == float('-inf'):
            return None
        return super().default(obj)

def json_dumps(obj):
    """
    Serialize obj to a JSON formatted string using the custom encoder.
    
    Args:
        obj: The object to serialize
        
    Returns:
        A JSON formatted string
    """
    return json.dumps(obj, cls=CustomJSONEncoder)

def json_loads(s):
    """
    Deserialize s (a str, bytes or bytearray instance containing a JSON document)
    to a Python object.
    
    Args:
        s: The JSON string to deserialize
        
    Returns:
        A Python object
    """
    return json.loads(s)
