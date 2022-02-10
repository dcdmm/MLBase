from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)
todos = {}


class Todo1(Resource):
    def get(self):
        # Default to 200 OK
        return {'task': 'apple'}


class Todo2(Resource):
    def get(self):
        # Set the response code to 201
        return {'task': 'babana'}, 201

# curl http://127.0.0.1:5000/Todo1
api.add_resource(Todo1, '/Todo1')
# curl http://127.0.0.1:5000/Todo2
api.add_resource(Todo2, '/Todo2')

if __name__ == "__main__":
    app.run()
