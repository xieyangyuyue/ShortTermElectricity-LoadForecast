# from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
# from flask_sqlalchemy import SQLAlchemy
# from werkzeug.security import generate_password_hash, check_password_hash
# import os
# app = Flask(__name__)
# app.config['SECRET_KEY'] = os.urandom(24)

# # MySQL 数据库配置
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost:3306/product_management_db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

# # 用户模型
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     password_hash = db.Column(db.Text, nullable=False)

#     # def set_password(self, password):
#     #     self.password_hash = generate_password_hash(password)

#     # def check_password(self, password):
#     #     return check_password_hash(self.password_hash, password)

# # 初始化数据库
# @app.before_first_request
# def create_tables():
#     db.create_all()

# # 登录页面
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         data = request.get_json()
#         email = data.get('email')
#         password = data.get('password')

#         user = User.query.filter_by(email=email).first()

#         if user and user.check_password(password):
#             session['user_id'] = user.id
#             return jsonify({
#                 'success': True,
#                 'message': '登录成功！',
#                 'redirect': url_for('dashboard')
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'message': '登录失败，请检查邮箱和密码'
#             }), 401

#     return render_template('login.html')

# # 注册页面
# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         data = request.get_json()
#         username = data.get('username')
#         email = data.get('email')
#         password = data.get('password')

#         errors = []
#         if not username:
#             errors.append('用户名不能为空')
#         if not email:
#             errors.append('邮箱不能为空')
#         if not password:
#             errors.append('密码不能为空')

#         if User.query.filter_by(username=username).first():
#             errors.append('该用户名已被使用')
#         if User.query.filter_by(email=email).first():
#             errors.append('该邮箱已被使用')

#         if errors:
#             return jsonify({
#                 'success': False,
#                 'errors': errors
#             }), 400

#         new_user = User(username=username, email=email,password_hash=password)
#         # new_user.set_password(password)
#         db.session.add(new_user)
#         db.session.commit()

#         return jsonify({
#             'success': True,
#             'message': '注册成功，请登录',
#             'redirect': url_for('login')
#         })

#     return render_template('register.html')

# # 仪表盘
# @app.route('/dashboard')
# def dashboard():
#     if 'user_id' not in session:
#         if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
#             return jsonify({
#                 'success': False,
#                 'message': '请先登录',
#                 'redirect': url_for('login')
#             }), 401
#         else:
#             flash('请先登录', 'error')
#             return redirect(url_for('login'))

#     user = User.query.get(session['user_id'])
#     return render_template('dashboard.html', user=user)

# # 退出登录
# @app.route('/logout')
# def logout():
#     session.pop('user_id', None)
#     if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
#         return jsonify({
#             'success': True,
#             'message': '已成功退出登录',
#             'redirect': url_for('login')
#         })
#     else:
#         flash('已成功退出登录', 'success')
#         return redirect(url_for('login'))

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pymysql
from pymysql.cursors import DictCursor
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
import json

app = Flask(__name__)
CORS(app, supports_credentials=True, origins="http://localhost:3000")

app.config["JWT_SECRET_KEY"] = "cE6BBb18-79c4-1c7F-755F-8edfd494eFFD"  # 请替换为安全的密钥
jwt = JWTManager(app)

# 数据库配置
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'product_management_db',
    'port': 3306,
    'charset': 'utf8mb4'
}


def get_connection():
    return pymysql.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database'],
        port=db_config['port'],
        charset=db_config['charset'],
        cursorclass=DictCursor
    )


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.json.get('username')
        email = request.json.get('email')
        password = request.json.get('password')

        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = "INSERT INTO Admin (username, email, password) VALUES (%s, %s, %s)"
                    cursor.execute(sql, (username, email, password))
                    conn.commit()
            return jsonify({'success': True, 'message': '注册成功', 'redirect': url_for('login')})
        except Exception as e:
            return jsonify({'success': False, 'message': f'注册失败: {str(e)}'})

    return render_template('register.html')


@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM Admin WHERE username = %s AND password = %s"
                cursor.execute(sql, (username, password))
                admin = cursor.fetchone()

        if admin:
            # 生成 Token，将用户 ID 或用户名存入 Token
            access_token = create_access_token(identity=admin['id'])
            
            return jsonify({
                'success': True,
                'message': '登录成功',
                'token': access_token,  # 返回 Token
                'redirect': '/lstm'
            })
        else:
            return jsonify({'success': False, 'message': '用户名或密码错误'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'登录失败: {str(e)}'})

# 修正后的 API 路由
@app.route("/configs/<int:id>", methods=["GET"])
def get_configs_data(id):
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                sql = "SELECT config FROM model_configs WHERE id = %s"
                cursor.execute(sql, (id,))
                result = cursor.fetchone()
        
        if result:
            # ✅ config 是字符串，先转为字典再 jsonify
            config_dict = json.loads(result["config"])
            return jsonify(config_dict)
        else:
            return jsonify({"error": "未找到数据"}), 404

    except Exception as e:
        return jsonify({"error": f"数据库错误: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
# @app.route('/dashboard')
# def dashboard():
#     return render_template('index.html')



# # # 登出
# # @app.route('/logout')
# # def logout():
# #     session.pop('user', None)
# #     return redirect(url_for('login'))


# # 模型训练
# @app.route('/dashboard/trans')
# def trans():
#     return render_template('trans.html')

# # 模型测试
# @app.route('/dashboard/ceshi')
# def ceshi():
#     return render_template('ceshi.html')

# # 数据集
# @app.route('/dashboard/shujuji')
# def shujuji():
#     return render_template('shujuji.html')

# if __name__ == '__main__':
#     app.run(debug=True)
    