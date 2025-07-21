from flask import request, render_template, redirect, url_for, make_response
from service import (
    get_user_by_username, register_user, verify_password, create_jwt_token, decode_jwt_token
)

def register_routes(app):

    @app.route('/')
    def index():
        return redirect(url_for('login'))

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            if get_user_by_username(username):
                return render_template('register.html', message="Username already exists.")
            register_user(username, password)
            return redirect(url_for('login'))
        return render_template('register.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            user = get_user_by_username(username)
            if user and verify_password(user['password'], password):
                token = create_jwt_token(username, password)
                response = make_response(render_template('home.html', token=token, message="Login successful", username=username))
                # response = redirect(url_for('home', token=token, message="Login successful")) 
                response.set_cookie('access_token', token, httponly=True)
                return response
            return render_template('login.html', message="Invalid username or password.")
        return render_template('login.html')

    @app.route('/home')
    def home():
        token = request.cookies.get('access_token')
        data = decode_jwt_token(token)
        if not data:
            return redirect(url_for('login'))
        return render_template('home.html', username=data['username'])

    @app.route('/start-tracker')
    def start_tracker():
        token = request.cookies.get('access_token')
        data = decode_jwt_token(token)
        if not data:
            return redirect(url_for('login'))
        import os
        os.system('python app.py')
        return render_template('home.html', username=data['username'], message="Tracker started. Press 'Esc' to stop.")

    @app.route('/logout')
    def logout():
        response = redirect(url_for('login'))
        response.delete_cookie('access_token')
        return response
