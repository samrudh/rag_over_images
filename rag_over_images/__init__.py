import warnings

# Suppress Google API Python version warning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*You are using a Python version.*which Google will stop supporting.*"
)

# Suppress HuggingFace resume_download warning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*resume_download.*is deprecated.*"
)
