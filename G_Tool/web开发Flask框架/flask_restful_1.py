from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

todos = {}


class TodoSimple(Resource):
    def get(self, todo_id):
        """GET请求时调用"""
        return {todo_id: todos[todo_id]}

    def post(self, todo_id):
        """POST请求时调用"""
        todos[todo_id] = request.form['data']  # 从form表单获取
        return {todo_id: todos[todo_id]}


api.add_resource(TodoSimple, '/<string:todo_id>')  # todo_id表示变量

if __name__ == '__main__':
    app.run()

'''
curl http://127.0.0.1:5000/todo1 -d 'data=apple' -X POST
{
    "todo1": "apple"
}

curl http://127.0.0.1:5000/todo1
{
    "todo1": "apple"
}
'''