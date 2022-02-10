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
        print(request.form)  # The form parameters
        # url命令发送JSON数据
        # curl  -H "Content-Type: application/json" http://127.0.0.1:5000/todo1  -X POST -d "{\"data\":123}"
        # json里的双引号必须使用反斜杠转义
        print(request.json)  # The parsed JSON data if :attr:`mimetype` indicates JSON
        if request.json is None:
            todos[todo_id] = request.form['data']
            return {todo_id: todos[todo_id]}
        else:
            todos[todo_id] = request.json['data']
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
