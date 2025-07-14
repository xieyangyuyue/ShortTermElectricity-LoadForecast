from flask import Flask, render_template, request, jsonify
import time
import random
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# 模拟不同模型的预测结果数据
def generate_model_data(model_type):
    """生成或获取模型预测结果数据"""
    # 模拟数据加载延迟
    time.sleep(1.5)
    
    if model_type == 'lstm':
        return {
            'title': 'LSTM 模型预测结果',
            'metrics': {
                'mae': 0.032,
                'rmse': 0.051,
                'accuracy': 89
            },
            'feature_importance': {
                '温度': 0.32,
                '历史负荷': 0.28,
                '湿度': 0.15,
                '节假日': 0.08,
                '风速': 0.05
            },
            'image_url': '/static/images/lstm_result.png'  # 替换为实际图片路径
        }
    elif model_type == 'gru':
        # 生成模拟的负荷预测表格数据
        dates = [(datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M') for i in range(24, -1, -1)]
        actual_load = [random.uniform(1000, 1800) for _ in dates]
        forecast_load = [load * (1 + random.uniform(-0.05, 0.05)) for load in actual_load]
        
        return {
            'title': 'GRU 模型预测结果',
            'data': list(zip(dates, forecast_load, actual_load)),
            'columns': ['时间', '预测负荷', '实际负荷'],
            'performance': '训练时间比 LSTM 缩短 23%，极端天气下误差上升 15%'
        }
    elif model_type == 'xgboost':
        return {
            'title': 'XGBoost 模型预测结果',
            'feature_importance': {
                '温度': 4.2,
                '历史负荷': 3.8,
                '节假日': 1.5,
                '湿度': 2.1,
                '风速': 0.9
            },
            'performance': '工作日/周末模式区分度高 (F1-score=0.92)，适合趋势级预测'
        }
    elif model_type == 'lightgbm':
        return {
            'title': 'LightGBM 模型预测结果',
            'accuracy_rates': {
                '工作日': 92,
                '周末': 88,
                '夏季高峰': 85,
                '冬季高峰': 87
            },
            'performance': '大规模数据训练效率比 XGBoost 快 40%，小样本易过拟合'
        }
    else:
        return {'error': '模型类型不支持'}

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html', active_model='lstm')

@app.route('/api/model_result', methods=['GET'])
def get_model_result():
    """获取模型预测结果的 API 接口"""
    model_type = request.args.get('model_type', 'lstm')
    
    try:
        # 模拟加载过程
        time.sleep(1)
        result = generate_model_data(model_type)
        return jsonify({
            'success': True,
            'data': result,
            'message': '数据加载完成'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'加载失败: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)