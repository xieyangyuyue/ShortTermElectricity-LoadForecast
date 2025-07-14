import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager

class Database:
    def __init__(self, host, user, password, database, port=3306, charset='utf8mb4'):
        self.host = host
        self.user = user
        self.password = str(password)  # 确保密码是字符串类型
        self.database = database
        self.port = port
        self.charset = charset

    @contextmanager
    def get_connection(self):
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            port=self.port,
            charset=self.charset,
            cursorclass=DictCursor
        )
        try:
            yield conn
        finally:
            conn.close()

    # 管理员操作
    def create_admin(self, username, email, password_hash):
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                sql = "INSERT INTO Admin (username, email, password_hash) VALUES (%s, %s, %s)"
                cursor.execute(sql, (username, email, password_hash))
                conn.commit()
                return cursor.lastrowid

    def verify_admin(self, username, password_hash=""):
        """
        验证管理员
        如果提供了密码哈希，则验证用户名和密码
        如果没有提供密码哈希，则仅检查用户名是否存在
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                if password_hash:
                    sql = "SELECT COUNT(*) as count FROM Admin WHERE username = %s AND password_hash = %s"
                    cursor.execute(sql, (username, password_hash))
                else:
                    sql = "SELECT COUNT(*) as count FROM Admin WHERE username = %s"
                    cursor.execute(sql, (username,))
                    
                result = cursor.fetchone()
                return result['count'] > 0

    def get_admin_by_email(self, email):
        """通过邮箱查找管理员"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM Admin WHERE email = %s"
                cursor.execute(sql, (email,))
                return cursor.fetchone()

    def update_admin_email(self, username, new_email):
        """更新管理员邮箱"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                sql = "UPDATE Admin SET email = %s WHERE username = %s"
                cursor.execute(sql, (new_email, username))
                conn.commit()
                return cursor.rowcount > 0