"""
This file contains a single dictionary of server configuration and file paths. 
"""
import os
import datetime

ROOT = os.path.dirname(__file__)

config = {
    'Build': os.path.join(ROOT, 'client', 'build'), # Client built path.
    'Log': os.path.join(ROOT, 'log', f'log_{datetime.datetime.now().strftime("%Y%m%d")}.log') # Log file path.
}

