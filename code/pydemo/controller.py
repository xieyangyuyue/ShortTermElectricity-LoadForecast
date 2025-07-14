# from flask import Flask, request, jsonify, g
# from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
# from werkzeug.security import generate_password_hash, check_password_hash
# import pymysql
# from pymysql.cursors import DictCursor
# from pymysqlpool import ConnectionPool
# import traceback
# from flask_cors import CORS
# import os
# from flask import render_template,request
# import json

# # 初始化Flask应用
# app = Flask(__name__)
# app.config['SECRET_KEY'] = os.urandom(24)

# app = Flask(__name__)
# app.config['JWT_SECRET_KEY'] = 'your-secret-key-here'  # 生产环境应从配置文件或环境变量获取
# jwt = JWTManager(app)




# # 启用CORS
# CORS(app)  # 允许所有域名跨域访问

# # 数据库连接池配置
# DB_CONFIG = {
#     "host": "localhost",
#     "user": "root",
#     "password": "123456",
#     "database": "product_management_db",
#     "port": 3306,
#     "cursorclass": DictCursor,
#     "autocommit": True
# }

# # 创建数据库连接池
# pool = ConnectionPool(size=5, name='pool', **DB_CONFIG)

# # 数据库操作类
# class Database:
#     def __init__(self, pool):
#         self.pool = pool
    
#     def execute_query(self, query, params=None):
#         try:
#             with self.pool.connection() as conn:
#                 with conn.cursor() as cursor:
#                     cursor.execute(query, params)
#                     return cursor.fetchall()
#         except Exception as e:
#             traceback.print_exc()
#             raise e
    
#     def execute_update(self, query, params=None):
#         try:
#             with self.pool.connection() as conn:
#                 with conn.cursor() as cursor:
#                     cursor.execute(query, params)
#                     return cursor.lastrowid
#         except Exception as e:
#             traceback.print_exc()
#             raise e
    
#     def create_admin(self, username, password):
#         # 安全地哈希密码
#         password_hash = generate_password_hash(password)
#         query = "INSERT INTO admins (username, password_hash) VALUES (%s, %s)"
#         return self.execute_update(query, (username, password_hash))
    
#     def verify_admin(self, username, password):
#         query = "SELECT * FROM admins WHERE username = %s"
#         result = self.execute_query(query, (username,))
        
#         if not result:
#             return False
            
#         stored_hash = result[0]['password_hash']
#         return check_password_hash(stored_hash, password)

# # 在请求前初始化数据库连接
# @app.before_request
# def before_request():
#     g.db = Database(pool)

# @app.route('/register', methods=['POST'])
# def register():
#     data = request.get_json()
    
#     # 数据验证
#     required_fields = ['username', 'password']
#     for field in required_fields:
#         if field not in data:
#             return jsonify({"success": False, "message": f"缺少必要字段: {field}"}), 400
    
#     # 验证两次密码是否一致
#     if data['password'] != data['confirm_password']:
#         return jsonify({"success": False, "message": "两次输入的密码不一致"}), 400
    
#     # 密码强度验证（示例：至少8个字符，包含字母和数字）
#     if len(data['password']) < 8:
#         return jsonify({"success": False, "message": "密码长度至少为8个字符"}), 400
#     if not any(char.isalpha() for char in data['password']):
#         return jsonify({"success": False, "message": "密码必须包含字母"}), 400
#     if not any(char.isdigit() for char in data['password']):
#         return jsonify({"success": False, "message": "密码必须包含数字"}), 400
    
#     # 生成密码哈希
#     password_hash = generate_password_hash(data['password'])
    
#     try:
        
#         admin_id = g.db.create_admin(
#             username=data['username'],
#             password_hash=password_hash  # 传递哈希后的密码
#         )
#         return jsonify({"success": True, "admin_id": admin_id})
#     except pymysql.IntegrityError as e:
#         if 'Duplicate entry' in str(e):
#             return jsonify({"success": False, "message": "用户名已存在"}), 400
#         return jsonify({"success": False, "message": "数据库错误"}), 500
#     except Exception as e:
#         app.logger.error(f"注册失败: {str(e)}")  # 记录详细错误日志
#         return jsonify({"success": False, "message": "注册失败，请重试"}), 500
    
# # 登录API
# @app.route('/login', methods=['POST'])
# def login():
#     data = request.get_json()
    
#     # 数据验证
#     required_fields = ['username', 'password']
#     for field in required_fields:
#         if field not in data:
#             return jsonify({"success": False, "message": f"缺少必要字段: {field}"}), 400
    
#     try:
#         is_valid = g.db.verify_admin(
#             username=data['username'],
#             password=data['password']
#         )
        
#         if is_valid:
#             # 生成JWT令牌
#             access_token = create_access_token(identity=data['username'])
#             return jsonify({
#                 "success": True, 
#                 "access_token": access_token,
#                 "token_type": "bearer"
#             })
#         else:
#             return jsonify({"success": False, "message": "用户名或密码错误"}), 401
    
#     except Exception as e:
#         return jsonify({"success": False, "message": "登录失败，请重试"}), 500

# # 受保护的API示例
# @app.route('/api/admin/profile', methods=['GET'])
# @jwt_required()
# def get_profile():
#     current_user = get_jwt_identity()
#     return jsonify({"success": True, "username": current_user})

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify, g
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import pymysql
from pymysql.cursors import DictCursor
from pymysqlpool import ConnectionPool
import traceback
from flask_cors import CORS
import os

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['JWT_SECRET_KEY'] = 'your-secret-key-here'
jwt = JWTManager(app)

# 启用CORS
CORS(app)

# 数据库连接池配置
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "product_management_db",
    "port": 3306,
    "cursorclass": DictCursor,
    "autocommit": True
}

# 创建数据库连接池
pool = ConnectionPool(size=5, name='pool', **DB_CONFIG)

# 数据库操作类
class Database:
    def __init__(self, pool):
        self.pool = pool
    
    def execute_query(self, query, params=None):
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    return cursor.fetchall()
        except Exception as e:
            traceback.print_exc()
            raise e
    
    def execute_update(self, query, params=None):
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    return cursor.lastrowid
        except Exception as e:
            traceback.print_exc()
            raise e
    
    def create_admin(self, username, password):
        # 直接存储原始密码（不推荐！仅用于测试）
        query = "INSERT INTO admins (username, password_hash) VALUES (%s, %s)"
        return self.execute_update(query, (username, password))
    
    def verify_admin(self, username, password):
        # 直接比较原始密码（不推荐！仅用于测试）
        query = "SELECT * FROM admins WHERE username = %s AND password_hash = %s"
        result = self.execute_query(query, (username, password))
        return len(result) > 0

# 在请求前初始化数据库连接
@app.before_request
def before_request():
    g.db = Database(pool)

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    
    # 数据验证
    required_fields = ['username', 'password']
    for field in required_fields:
        if field not in data:
            return jsonify({"success": False, "message": f"缺少必要字段: {field}"}), 400
    
    # 验证两次密码是否一致
    if data.get('confirm_password') and data['password'] != data['confirm_password']:
        return jsonify({"success": False, "message": "两次输入的密码不一致"}), 400
    
    # 密码强度验证
    if len(data['password']) < 8:
        return jsonify({"success": False, "message": "密码长度至少为8个字符"}), 400
    if not any(char.isalpha() for char in data['password']):
        return jsonify({"success": False, "message": "密码必须包含字母"}), 400
    if not any(char.isdigit() for char in data['password']):
        return jsonify({"success": False, "message": "密码必须包含数字"}), 400
    
    try:
        # 直接使用原始密码
        admin_id = g.db.create_admin(
            username=data['username'],
            password=data['password']
        )
        return jsonify({"success": True, "admin_id": admin_id})
    except pymysql.IntegrityError as e:
        if 'Duplicate entry' in str(e):
            return jsonify({"success": False, "message": "用户名已存在"}), 400
        return jsonify({"success": False, "message": "数据库错误"}), 500
    except Exception as e:
        app.logger.error(f"注册失败: {str(e)}")
        return jsonify({"success": False, "message": "注册失败，请重试"}), 500
    
# 登录API
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    # 数据验证
    required_fields = ['username', 'password']
    for field in required_fields:
        if field not in data:
            return jsonify({"success": False, "message": f"缺少必要字段: {field}"}), 400
    
    try:
        # 直接验证原始密码
        is_valid = g.db.verify_admin(
            username=data['username'],
            password=data['password']
        )
        
        if is_valid:
            # 生成JWT令牌
            access_token = create_access_token(identity=data['username'])
            return jsonify({
                "success": True, 
                "access_token": access_token,
                "token_type": "bearer"
            })
        else:
            return jsonify({"success": False, "message": "用户名或密码错误"}), 401
    
    except Exception as e:
        return jsonify({"success": False, "message": "登录失败，请重试"}), 500

# 受保护的API示例
@app.route('/api/admin/profile', methods=['GET'])
@jwt_required()
def get_profile():
    current_user = get_jwt_identity()
    return jsonify({"success": True, "username": current_user})

if __name__ == '__main__':
    app.run(debug=True)