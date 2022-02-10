from flask import Flask

# 实例化
app = Flask(__name__)


# http://127.0.0.1:5000/duan
# 接受'GET'与'POST'请求
@app.route("/duan", methods=['GET', 'POST'])
def hello_world():
    return "<p>Hello, World!</p>"


# http://127.0.0.1:/chao
# 只接受'POST'请求
@app.route("/chao", methods=['POST'])
def hello_python():
    return "<p>Hello, Python!</p>"


if __name__ == '__main__':
    app.run()
