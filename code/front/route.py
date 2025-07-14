from flask import Flask, request

app = Flask(__name__)

# 基本路由
@app.route('/')
def index():
    return '欢迎访问首页！'

# 带参数的路由
@app.route('/user/<username>')
def show_user(username):
    return f'用户: {username}'

# 带类型限定的参数
@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f'文章 ID: {post_id}'

# 指定 HTTP 方法
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return '处理登录表单'
    else:
        return '显示登录页面'

if __name__ == '__main__':
    app.run()