from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pymysql
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/configs/{id}")
def get_configs_data(id: int):
    conn = pymysql.connect(
        host='localhost', user='root', password='123456', database='product_management_db'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT config FROM model_configs WHERE id=%s", (id,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if result:
        return json.loads(result[0])  # 确保返回格式为 JSON 对象
    else:
        return {"error": "未找到数据"}
