import jwt
import time
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from flask import current_app as app

mongo = PyMongo()

def mongo_hehe(app):
    mongo.init_app(app)

def hash_password(password):
    return generate_password_hash(password)

def verify_password(stored_password, provided_password):
    return check_password_hash(stored_password, provided_password)

def get_user_by_username(username):
    return mongo.db.login_cred.find_one({'username': username})

def register_user(username, password):
    mongo.db.login_cred.insert_one({
        'username': username,
        'password': hash_password(password)
    })

def create_jwt_token(username, password):
    payload = {
        'username': username,
        'password': password,  
        'expiretime': int(time.time()) + 3600
    }
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    return token

def decode_jwt_token(token):
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None