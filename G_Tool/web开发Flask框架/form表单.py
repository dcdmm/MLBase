from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/index', methods=['GET', 'POST'])
def index():
    # request.method:The method the request was made with, such as GET.
    if request.method == 'GET':
        # Renders a template from the template folder with the given context.
        return render_template('example.html')
    if request.method == 'POST':
        name = request.form.get('name_') # 从form表单获取name_属性
        password = request.form.get('password_')  # 从form表单获取password_属性
        print(name, password)
        return "this is post"


if __name__ == '__main__':
    app.run()
