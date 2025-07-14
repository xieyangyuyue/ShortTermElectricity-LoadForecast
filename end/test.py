from db_operations import Database  
import pymysql

DB_CONFIG = {
    "host": "localhost",       
    "user": "root",            
    "password": "123456",      
    "database": "product_management_db",  
    "port": 3306               
}

db = Database(**DB_CONFIG)

try:
    admin_id = db.create_admin(username="zsc1", password_hash="123456",email="111@qq.com")
    print(f"创建管理员成功，ID：{admin_id}")
    # 再查询刚插入的管理员，看看是否能查到
  

    is_valid = db.verify_admin(username="test_admin", password_hash="hashed_password_123")
    print(f"管理员验证结果：{'成功' if is_valid else '失败'}")

except pymysql.MySQLError as db_err:
    print(f"数据库操作错误：{db_err}，错误代码：{db_err.args[0]}")
except TypeError as type_err:
    print(f"类型错误：{type_err}")
except Exception as e:
    print(f"其他未知错误：{e}")

