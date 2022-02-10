from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

todos = {}


class TodoSimple(Resource):
    def post(self):
        """POST请求时调用"""
        parser = reqparse.RequestParser()
        parser.add_argument('width', type=int, help='width!!!')  # 类似命令行解析argparse模块
        parser.add_argument('hight', type=int, help='hight!!!')
        args = parser.parse_args()
        print(args)
        area = args.get('width') * args.get('hight')
        return area


# curl http://127.0.0.1:5000/index -d 'width=3' -d 'hight=4' -X POST
api.add_resource(TodoSimple, '/index')

if __name__ == '__main__':
    app.run()
