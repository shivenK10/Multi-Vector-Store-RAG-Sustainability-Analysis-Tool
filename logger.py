import logging
from logging.handlers import RotatingFileHandler
from typing import Literal

class Logger:
    def __init__(self, name: str, log_file_needed: bool = False, log_file: str = '', level: Literal['DEV', 'PROD'] = 'DEV'):
        """Initialize logger with console and file output"""
        
        # Validate that log_file is provided when log_file_needed is True
        if log_file_needed and not log_file.strip():
            raise ValueError("A file name is required when log_file_needed is set to True")
        
        self.name = name
        self.log_file_needed = log_file_needed
        self.log_file = log_file
        
        self.logger = logging.getLogger(name)
        if level.upper() == 'DEV':
            self.logger.setLevel(logging.DEBUG)
        elif level.upper() == 'PROD':
            self.logger.setLevel(logging.INFO)
        else:
            raise ValueError("The value of level must be 'DEV' or 'PROD'")
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Rotating file handler with 5MB cap
        if self.log_file_needed:  
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=5*1024*1024,  # 5MB
                backupCount=5  # Keep 5 backup files
            )
            file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        if self.log_file_needed:
            file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        if self.log_file_needed:
            self.logger.addHandler(file_handler)

    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)

