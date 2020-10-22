"""
This file contains server exceptions inherited from werkzeug.
"""
from werkzeug.exceptions import BadRequest


class BadArgumentRequest(BadRequest):
    def __init__(self, msg=''):
        self.msg = msg
        self.code = 400


class UnknownAPIRequest(BadRequest):
    def __init__(self, msg=''):
        self.msg = msg
        self.code = 400