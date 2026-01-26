"""
Model configuration settings for GlobalHealthAtlas
"""
# Model parameters
MAX_MODEL_LEN = 40960
BATCH_SIZE = 4000  # Process 4000 items at a time to prevent crashes
SAFE_INPUT_LIMIT = MAX_MODEL_LEN - 1024  # Leave 1024 tokens for output

# Stop token IDs
STOP_TOKEN_IDS = [151643, 151645]

# JSON schema for validation
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "Accuracy": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "description": {"type": "string"}
            },
            "required": ["score", "description"]
        },
        "Reasoning": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "description": {"type": "string"}
            },
            "required": ["score", "description"]
        },
        "Completeness": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "description": {"type": "string"}
            },
            "required": ["score", "description"]
        },
        "Consensus Alignment": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "description": {"type": "string"}
            },
            "required": ["score", "description"]
        },
        "Terminology Norms": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "description": {"type": "string"}
            },
            "required": ["score", "description"]
        },
        "Insightfulness": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "description": {"type": "string"}
            },
            "required": ["score", "description"]
        }
    },
    "required": ["Accuracy", "Reasoning", "Completeness", "Consensus Alignment", "Terminology Norms", "Insightfulness"]
}