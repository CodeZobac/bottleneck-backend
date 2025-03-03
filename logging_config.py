import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(log_level=logging.INFO, log_file=None):
    """Configure logging for the application."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Always add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
    
    # Set more detailed logging for specific modules
    logging.getLogger('component_handler').setLevel(logging.DEBUG)
    # logging.getLogger('model').setLevel(logging.DEBUG)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('pandas').setLevel(logging.WARNING)
    
    return root_logger
