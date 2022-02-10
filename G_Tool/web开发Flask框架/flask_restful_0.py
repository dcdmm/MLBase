from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

# curl http://127.0.0.1:5000/index
api.add_resource(HelloWorld, '/index')

if __name__ == '__main__':
    app.run()
