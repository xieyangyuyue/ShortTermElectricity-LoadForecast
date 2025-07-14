import json
import pymysql

# 读取 JSON 文件
with open(r'D:\pythondemo\project-training-2-master\results\load_forecast_5days.json', 'r') as f:
    config_data = json.load(f)

# 连接数据库
conn = pymysql.connect(
    host='localhost', user='root', password='123456', database='product_management_db', charset='utf8mb4'
)
cursor = conn.cursor()

# 插入 JSON 数据
sql = "INSERT INTO model_configs (name, config) VALUES (%s, %s)"
cursor.execute(sql, ('Transformer配置', json.dumps(config_data)))
conn.commit()

cursor.close()
conn.close()
