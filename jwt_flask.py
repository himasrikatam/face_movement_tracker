# JSON WEB TOKEN (JWT) for Flask
# no need to store session data on the server
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv
from flask_jwt_extended import (
    JWTManager,
    jwt_required,
    create_access_token,
    get_jwt_identity,
    set_access_cookies,
    unset_jwt_cookies
)
load_dotenv()

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = os.getenv('SECRET_KEY') # required because without this flask-jwt-extended will not work
app.config["MONGO_URI"] = os.getenv('MONGO_URI')
mongo = PyMongo(app)
jwt = JWTManager(app)
app.config.update({
    "JWT_TOKEN_LOCATION": ["cookies"],  # Uses cookies instead of headers 
    "JWT_COOKIE_HTTPONLY": True,  # Sets HTTP-only flag for security (prevents JavaScript access)
    "JWT_COOKIE_SECURE": False,  # Set to True in production with HTTPS - not required in dev(localhost) as it doesnt use HTTPS but in prod it uses https 
    # setting to false in prod can lead to security issues
})

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET','POST']) # GET (shows form)
def register():
    if request.method == 'POST':
     username = request.form['username']
     password = request.form['password']
     if mongo.db.login_cred.find_one({'username': username}):
         return render_template('register.html', message="Username already exists.")
     mongo.db.login_cred.insert_one({
        'username': username,
        'password': generate_password_hash(password)
     })
     return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = mongo.db.login_cred.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            access_token = create_access_token(identity=username)
            response = make_response(
                render_template('home.html', 
                    token=access_token, 
                    message="Login successful. Click 'Start Tracker' to begin.")
                    )
            set_access_cookies(response, access_token)
            return response
            # return  render_template('home.html', token=access_token, message="Login successful. Click 'Start Tracker' to begin.")
        return render_template('login.html', message="Invalid username or password.")
    return render_template('login.html')

@app.route('/home')
@jwt_required()
def home(): 
    current_user = get_jwt_identity()
    os.system('python app.py')  
    return render_template('home.html', username=current_user)

@app.route('/start-tracker')
@jwt_required()
def start_tracker():
    current_user = get_jwt_identity()
    os.system('python app.py')  
    return render_template('home.html', username=current_user, message="Face movement tracker started. Click 'esc' to stop.")

@app.route('/logout')
@jwt_required()
# def logout():
#     current_user = get_jwt_identity()
#     return redirect(url_for('login'))
def logout():
    response = redirect(url_for('login')) # This IS a Response object, render_template returns a string(html) which needs make_response to convert it to a Response object
    unset_jwt_cookies(response)
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5050)

# cookies => client side storage - small text files stored in the browser
# session => server side storage - stores data on the server
# headers => used to send additional information with the request/response Sent with every HTTP request/response
# ┌───────────────────────────────────────────────────────────────────────┐
# │                        JWT Authentication Flow                        │
# └───────────────────────────────────────────────────────────────────────┘
#                                │
#                                ▼
#                       ┌─────────────────┐
#                       │    /register    │
#                       │ (POST)          │
#                       └─────────────────┘
#                                │
#                                ▼
#                 ┌──────────────┴──────────────┐
#                 │ 1. Hash password            │
#                 │ 2. Store user in MongoDB    │
#                 └──────────────┬──────────────┘
#                                │
#                                ▼
#                       ┌─────────────────┐
#                       │    /login       │
#                       │ (POST)          │
#                       └─────────────────┘
#                                │
#                                ▼
#                 ┌──────────────┴──────────────┐
#                 │ 1. Verify credentials       │
#                 │ 2. Create JWT               │
#                 │ 3. Set HTTP-only cookie    │
#                 └──────────────┬──────────────┘
#                                │
#                                ▼
#                       ┌─────────────────┐
#                       │ Protected Routes │
#                       │ (/home,          │
#                       │ /start-tracker)  │
#                       └─────────────────┘
#                                │
#                                ▼
#                 ┌──────────────┴──────────────┐
#                 │ 1. Verify JWT in cookie     │
#                 │ 2. Get user identity         │
#                 └──────────────┬──────────────┘
#                                │
#                                ▼
#                       ┌─────────────────┐
#                       │    /logout      │
#                       └─────────────────┘
#                                │
#                                ▼
#                       ┌─────────────────┐
#                       │ Clear JWT cookie │
#                       └─────────────────┘