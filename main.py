from flask import Flask
from controller import register_routes
from service import mongo_hehe
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config["MONGO_URI"] = os.getenv('MONGO_URI')

mongo_hehe(app)
register_routes(app)

if __name__ == '__main__':
    app.run(debug=True, port=5050)
