from flask import Flask, jsonify

app = Flask(__name__)
# Serialize objects to ASCII-encoded JSON. If this is disabled, the JSON returned from jsonify will contain Unicode characters.
app.config['JSON_AS_ASCII'] = False


@app.route('/index')
def index():
    data = {
        'name': '张三'
    }
    # Serialize data to JSON and wrap it in a :class:`~flask.Response
    return jsonify(code=200, status=0, message='ok', value=data)


if __name__ == '__main__':
    app.run()
