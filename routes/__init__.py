# Do not format this file, import order matters, save without formatting (ctrl + k, or ctrl + shift + p and search for save without formatting)
from flask import Blueprint

open_bp = Blueprint('open', __name__)

from . import open