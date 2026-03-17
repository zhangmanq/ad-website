"""
AD辅助诊断后端API - 仅随机森林版（支持异步任务与SSE进度推送）
修正：文件对象在请求结束后关闭的问题
"""
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import sys
import joblib
import numpy as np
import nibabel as nib
import tempfile
import datetime
import traceback
import shutil
import threading
import uuid
import queue
import json
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

# 导入AI建议生成器
from my_ai_clients import generate_clinical_advice

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入特征提取器
try:
    from feature_extractor import MRIFeatureExtractorV2
    print("✅ 成功导入特征提取器V2")
except ImportError as e:
    print(f"❌ 导入特征提取器V2失败: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# ================ 配置 ================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'nii', 'nii.gz', 'img', 'hdr'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"✅ 上传目录: {UPLOAD_FOLDER}")

# ================ 全局任务存储 ================
tasks = {}
tasks_lock = threading.Lock()

# ================ 加载训练好的模型 ================
print("\n" + "="*60)
print("AD辅助诊断系统 - 仅随机森林版 (异步进度版)")
print("="*60)

rf_model = None
scaler = None
selected_features = []
class_labels = {}
feature_extractor = None
gender_encoder = None
model_accuracy = 0.9725

try:
    MODEL_DIR = r'E:\model修改3'   # 请根据实际情况修改
    
    RF_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
    SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
    FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')
    LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')
    INFO_PATH = os.path.join(MODEL_DIR, 'training_info.pkl')
    
    if os.path.exists(FEATURE_NAMES_PATH):
        selected_features = joblib.load(FEATURE_NAMES_PATH)
        print(f"✅ 从单独文件加载特征名称，共 {len(selected_features)} 个")
    
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("✅ 从单独文件加载标准化器")
    
    if os.path.exists(RF_PATH):
        rf_model = joblib.load(RF_PATH)
        print("✅ 从单独文件加载随机森林模型")
    
    # 后备：加载Web应用模型包
    if rf_model is None or scaler is None or not selected_features:
        WEB_MODEL_PATH = os.path.join(MODEL_DIR, 'web_ad_model.pkl')
        if os.path.exists(WEB_MODEL_PATH):
            print("部分单独文件缺失，尝试加载Web应用模型包...")
            web_model = joblib.load(WEB_MODEL_PATH)
            print(f"Web模型包中的键: {list(web_model.keys())}")
            
            if rf_model is None:
                rf_model = web_model.get('rf_model')
            if scaler is None:
                scaler = web_model.get('scaler')
            if not selected_features:
                selected_features = web_model.get('selected_features', [])
            class_labels = web_model.get('class_labels', {0: 'Demented', 1: 'Nondemented', 2: 'Converted'})
            
            print("✅ 从Web模型包补充加载完成")
    
    if not class_labels:
        if os.path.exists(INFO_PATH):
            info = joblib.load(INFO_PATH)
            class_labels = info.get('class_labels', {0: 'Demented', 1: 'Nondemented', 2: 'Converted'})
        else:
            class_labels = {0: 'Demented', 1: 'Nondemented', 2: 'Converted'}
    
    if os.path.exists(LABEL_ENCODERS_PATH):
        label_encoders = joblib.load(LABEL_ENCODERS_PATH)
        gender_encoder = label_encoders.get('Clinical_M/F', None)
        print(f"✅ 加载性别编码器: {gender_encoder}")
    
    if os.path.exists(INFO_PATH):
        info = joblib.load(INFO_PATH)
        perf = info.get('model_performance', {})
        if 'Random Forest' in perf:
            model_accuracy = perf['Random Forest'].get('mean_val_accuracy', 0.9725)
        print(f"✅ 模型准确率: {model_accuracy:.4f}")
    
    COMMON_MASK_PATH = r'E:\PROJECT_FINALLY\OAS2_OVERALL_group_mask.nii.gz'
    if not os.path.exists(COMMON_MASK_PATH):
        COMMON_MASK_PATH = os.path.join(BASE_DIR, 'OAS2_OVERALL_group_mask.nii.gz')
    
    if os.path.exists(COMMON_MASK_PATH):
        print(f"✅ 找到公共掩膜: {COMMON_MASK_PATH}")
    else:
        COMMON_MASK_PATH = None
        print("⚠️ 未找到公共掩膜，将使用全脑提取")
    
    feature_extractor = MRIFeatureExtractorV2(common_mask_path=COMMON_MASK_PATH)
    print("✅ 特征提取器初始化成功")
    
except Exception as e:
    print(f"❌ 模型/特征提取器初始化失败: {e}")
    traceback.print_exc()

# ================ 模型自检 ================
print("\n" + "="*60)
print("启动模型自检...")
if rf_model is not None and scaler is not None and selected_features:
    try:
        np.random.seed(42)
        X1 = np.random.randn(1, len(selected_features))
        X2 = np.random.randn(1, len(selected_features))
        X1_scaled = scaler.transform(X1)
        X2_scaled = scaler.transform(X2)
        pred1 = rf_model.predict(X1_scaled)[0]
        pred2 = rf_model.predict(X2_scaled)[0]
        prob1 = rf_model.predict_proba(X1_scaled)[0]
        prob2 = rf_model.predict_proba(X2_scaled)[0]
        print("随机森林自检：")
        print(f"  预测类别: {pred1} vs {pred2}")
        print(f"  概率示例: {prob1[0]:.4f} vs {prob2[0]:.4f}")
        if pred1 == pred2 and np.allclose(prob1, prob2):
            print("⚠️ 警告：随机森林对两个随机样本输出相同预测！可能模型有问题。")
        else:
            print("✅ 随机森林模型自检通过。")
    except Exception as e:
        print(f"⚠️ 随机森林自检过程中出现异常: {e}")
else:
    print("随机森林模型未加载，跳过自检。")
print("="*60 + "\n")

# ================ 辅助函数（保持不变） ================
def allowed_file(filename):
    if not filename:
        return False
    if filename.lower().endswith('.nii.gz'):
        return True
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def encode_gender(gender_str):
    if not gender_str:
        return 0.0
    g = gender_str.lower().strip()
    if gender_encoder is not None:
        if g in ['male', 'm', '男']:
            raw = 'M'
        elif g in ['female', 'f', '女']:
            raw = 'F'
        else:
            raw = 'M'
        try:
            code = gender_encoder.transform([raw])[0]
            return float(code)
        except:
            return 0.0
    else:
        return 1.0 if g in ['female', 'f', '女'] else 0.0

def generate_diagnosis_description(label, probability):
    desc_map = {
        'Demented': f"患者表现出明显的阿尔兹海默症迹象，基于MRI特征分析，诊断概率为{probability:.1%}。建议立即进行临床评估和干预。",
        'Nondemented': f"认知功能正常，无显著阿尔兹海默症迹象，基于MRI特征分析，诊断概率为{probability:.1%}。建议保持定期随访。",
        'Converted': f"处于早期认知障碍阶段，有转化为阿尔兹海默症的风险，基于MRI特征分析，诊断概率为{probability:.1%}。建议加强监测和预防干预。"
    }
    return desc_map.get(label, f"AI模型分析完成")

def generate_recommendation(label, age):
    if label == 'Demented':
        return {
            'title': '临床建议 - 痴呆阶段',
            'age_note': f'患者年龄{age}岁，处于阿尔兹海默症高风险阶段' if age else None,
            'actions': [
                '立即预约神经科专科门诊',
                '进行全面的神经心理学评估',
                '考虑药物治疗方案（如胆碱酯酶抑制剂）',
                '安排家庭护理支持计划',
                '每3-6个月进行一次MRI复查'
            ],
            'tags': ['紧急处理', '专科就诊', '药物治疗', '家庭护理']
        }
    elif label == 'Nondemented':
        return {
            'title': '临床建议 - 认知正常',
            'age_note': f'患者年龄{age}岁，认知功能保持良好' if age else None,
            'actions': [
                '建议每年进行一次认知功能筛查',
                '保持健康生活方式（地中海饮食、适度运动）',
                '参与认知训练活动（阅读、棋类等）',
                '管理心血管危险因素（血压、血糖、血脂）',
                '每年进行一次MRI基线复查'
            ],
            'tags': ['定期随访', '健康生活方式', '认知训练', '风险管理']
        }
    else:
        return {
            'title': '临床建议 - 转化期（早期认知障碍）',
            'age_note': f'患者年龄{age}岁，需密切监测认知变化' if age else None,
            'actions': [
                '建议3-6个月后复查MRI',
                '进行详细的神经心理学评估',
                '考虑预防性生活方式干预',
                '监测主观认知下降症状',
                '每6个月进行一次认知功能评估'
            ],
            'tags': ['密切监测', '早期干预', '定期评估', '预防措施']
        }

def generate_key_insights(label, probability):
    if label == 'Demented':
        return [
            f'诊断概率高达{probability:.1%}，强烈提示阿尔兹海默症',
            '海马体积明显萎缩，符合典型AD模式',
            '白质完整性下降，提示神经退行性变',
            '建议立即启动综合治疗计划'
        ]
    elif label == 'Nondemented':
        return [
            f'诊断概率为{probability:.1%}，认知功能保持良好',
            '关键脑区结构未见显著异常',
            '白质完整性保持良好',
            '继续维持健康生活方式'
        ]
    else:
        return [
            f'诊断概率为{probability:.1%}，处于转化风险期',
            '部分脑区出现早期萎缩迹象',
            '需密切监测认知功能变化',
            '建议启动预防性干预措施'
        ]

def get_top_features(features_dict, n=6):
    important_keys = [
        'Right_Hippocampus_Volume',
        'Left_Hippocampus_Kurtosis',
        'Right_Cerebral_White_Matter_Q25',
        'Left_Cerebral_White_Matter_StdIntensity',
        'Shape_Compactness',
        'Shape_EllipsoidDiameter_X'
    ]
    top_features = []
    for key in important_keys:
        if key in features_dict:
            value = features_dict[key]
            if 'Volume' in key and value < 2000:
                status = '异常'
            elif 'Kurtosis' in key and abs(value) > 5:
                status = '异常'
            elif 'Q25' in key and value < 50:
                status = '异常'
            elif 'Std' in key and value > 100:
                status = '异常'
            else:
                status = '正常'
            top_features.append({
                'name': key.replace('_', ' '),
                'value': f"{value:.2f}",
                'status': status
            })
        else:
            top_features.append({
                'name': key.replace('_', ' '),
                'value': 'N/A',
                'status': '未知'
            })
    return top_features[:n]

import re

def clean_markdown(text):
    if not text:
        return text
    text = re.sub(r'^[\s\u00A0\u200b\u200c\u200d\uFEFF]+', '', text)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        cleaned_lines.append(stripped if stripped != '' else '')
    for i, line in enumerate(cleaned_lines):
        if re.match(r'^#{1,6}\s+', line):
            cleaned_lines[i] = re.sub(r'^#{1,6}\s+', '', line)
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'__(.*?)__', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*(?!\s)(.*?)(?<!\s)\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_(?!\s)(.*?)(?<!\s)_', r'\1', text, flags=re.DOTALL)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if re.match(r'^\s*[-*]\s+', line):
            lines[i] = re.sub(r'^\s*[-*]\s+', '• ', line)
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ================ 后台任务处理函数 ================
def process_task(task_id, saved_paths, temp_upload_dir, age, gender, ai_model):
    """
    后台处理任务：
    saved_paths: 已保存的文件路径列表
    temp_upload_dir: 临时目录，处理完后需要删除
    """
    with tasks_lock:
        task_info = tasks.get(task_id)
        if not task_info:
            print(f"任务 {task_id} 不存在，终止处理")
            return
        q = task_info['queue']

    def send_progress(step, message, status='processing'):
        try:
            q.put_nowait({
                'step': step,
                'message': message,
                'status': status
            })
        except Exception as e:
            print(f"发送进度消息失败: {e}")

    try:
        send_progress('start', '开始处理', 'processing')

        # ---------- 步骤1：文件已保存（在调用线程中完成），直接确认 ----------
        send_progress('save', '文件保存完成', 'complete')

        # 主图像路径（取第一个文件）
        main_img_path = saved_paths[0]

        # ---------- 步骤2：格式检查与转换 ----------
        send_progress('convert', '正在检查并转换文件格式...', 'processing')
        converted_path = main_img_path
        is_temp_converted = False
        if main_img_path.lower().endswith('.hdr'):
            send_progress('convert', '检测到 Analyze 格式，正在转换为 NIfTI...', 'processing')
            try:
                converted_path, is_temp_converted = feature_extractor._check_and_convert_format(main_img_path)
                if is_temp_converted:
                    # 将转换后的文件也加入saved_paths以便清理
                    saved_paths.append(converted_path)
                send_progress('convert', '格式转换完成', 'complete')
            except Exception as e:
                send_progress('convert', f'格式转换失败: {str(e)}', 'error')
                raise
        else:
            send_progress('convert', '格式检查通过', 'complete')

        # ---------- 步骤3：生成个体掩膜 ----------
        send_progress('mask', '正在生成个体脑掩膜...', 'processing')
        try:
            img_nib = nib.load(converted_path)
            mask_img = feature_extractor._generate_individual_mask(img_nib)
            mask_path = os.path.join(temp_upload_dir, 'individual_mask.nii.gz')
            nib.save(mask_img, mask_path)
            send_progress('mask', '个体掩膜生成完成', 'complete')
        except Exception as e:
            send_progress('mask', f'个体掩膜生成失败: {str(e)}', 'error')
            raise

        # ---------- 步骤4：提取特征 ----------
        send_progress('feature', '正在提取多尺度特征...', 'processing')
        try:
            feature_values = feature_extractor.extract_selected_features(
                img_path=converted_path,
                selected_features_list=selected_features,
                subject_id=os.path.basename(main_img_path),
                age=age,
                gender=encode_gender(gender),
                mask_path=mask_path
            )
            if not feature_values:
                raise ValueError("特征提取返回空列表")
            features_dict = dict(zip(selected_features, feature_values))
            send_progress('feature', f'特征提取完成，共 {len(feature_values)} 个特征', 'complete')
        except Exception as e:
            send_progress('feature', f'特征提取失败: {str(e)}', 'error')
            raise

        # ---------- 步骤5：标准化与预测 ----------
        send_progress('predict', '正在运行随机森林模型...', 'processing')
        try:
            X = np.array(feature_values).reshape(1, -1)
            if scaler:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            pred_class = int(rf_model.predict(X_scaled)[0])
            proba = rf_model.predict_proba(X_scaled)[0].tolist()
            label_name = class_labels.get(pred_class, f'Class_{pred_class}')
            confidence = max(proba)
            send_progress('predict', f'预测完成：{label_name}（置信度{confidence:.1%}）', 'complete')
        except Exception as e:
            send_progress('predict', f'模型预测失败: {str(e)}', 'error')
            raise

        # ---------- 步骤6：生成AI建议（如果选择） ----------
        ai_advice = None
        if ai_model in ['deepseek', 'qwen2.5', 'llama3.2']:
            send_progress('ai', f'正在调用 {ai_model} 生成个性化建议...', 'processing')
            try:
                ai_advice = generate_clinical_advice(
                    ai_model=ai_model,
                    diagnosis_label=label_name,
                    probability=confidence,
                    age=age,
                    gender=gender
                )
                if ai_advice:
                    ai_advice = clean_markdown(ai_advice)
                    send_progress('ai', 'AI建议生成完成', 'complete')
                else:
                    send_progress('ai', 'AI建议生成失败（返回空），将使用规则建议', 'error')
            except Exception as e:
                send_progress('ai', f'AI建议生成异常: {str(e)}', 'error')
                ai_advice = None

        # ---------- 组装最终结果 ----------
        rule_recommendation = generate_recommendation(label_name, age)
        result = {
            'status': 'success',
            'data': {
                'diagnosis': {
                    'label': label_name,
                    'probability': confidence,
                    'description': generate_diagnosis_description(label_name, confidence),
                    'confidence': '高置信度' if confidence > 0.8 else '中等置信度' if confidence > 0.6 else '低置信度'
                },
                'probabilities': {
                    'Nondemented': proba[1] if len(proba) > 1 else 0.0,
                    'Demented': proba[0] if len(proba) > 0 else 0.0,
                    'Converted': proba[2] if len(proba) > 2 else 0.0
                },
                'top_features': get_top_features(features_dict),
                'recommendation': rule_recommendation,
                'ai_advice': ai_advice,
                'key_insights': generate_key_insights(label_name, confidence)
            },
            'technical_info': {
                'model_type': '随机森林',
                'model_accuracy': model_accuracy,
                'n_features': len(feature_values),
                'processed_file': os.path.basename(main_img_path),
                'ai_model_used': ai_model if ai_advice else None
            },
            'timestamp': datetime.datetime.now().isoformat()
        }

        with tasks_lock:
            tasks[task_id]['result'] = result
            tasks[task_id]['status'] = 'done'
        send_progress('done', '分析完成', 'complete')
        print(f"✅ 任务 {task_id} 处理完成")

    except Exception as e:
        print(f"❌ 任务 {task_id} 处理失败: {e}")
        traceback.print_exc()
        with tasks_lock:
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['error'] = str(e)
        send_progress('error', f'处理失败：{str(e)}', 'error')
    finally:
        # 清理临时目录
        if temp_upload_dir and os.path.exists(temp_upload_dir):
            try:
                shutil.rmtree(temp_upload_dir)
                print(f"✅ 清理临时目录: {temp_upload_dir}")
            except Exception as e:
                print(f"⚠️ 清理临时文件失败: {e}")

# ================ 新API路由 ================

@app.route('/api/start_predict', methods=['POST'])
def start_predict():
    """启动异步预测任务，返回task_id"""
    if not rf_model:
        return jsonify({'error': '随机森林模型未加载，无法进行预测'}), 500
    if not feature_extractor:
        return jsonify({'error': '特征提取器未初始化'}), 500
    if not selected_features:
        return jsonify({'error': '特征列表为空，无法提取特征'}), 500

    if 'files[]' not in request.files:
        return jsonify({'error': '没有上传图像文件（字段名应为 files[]）'}), 400
    
    files = request.files.getlist('files[]')
    if not files or len(files) == 0:
        return jsonify({'error': '文件列表为空'}), 400

    age = request.form.get('age', type=float, default=0)
    gender = request.form.get('gender', default='male')
    ai_model = request.form.get('ai_model', '').lower()

    # 生成唯一任务ID
    task_id = str(uuid.uuid4())

    # 立即将所有文件保存到临时目录，避免请求结束后文件流关闭
    temp_upload_dir = tempfile.mkdtemp(dir=UPLOAD_FOLDER)
    saved_paths = []
    try:
        for f in files:
            if f.filename:
                fname = secure_filename(f.filename)
                fpath = os.path.join(temp_upload_dir, fname)
                f.save(fpath)
                saved_paths.append(fpath)
                print(f"✅ 文件已保存到临时目录: {fpath}")
    except Exception as e:
        # 保存失败时清理临时目录
        shutil.rmtree(temp_upload_dir, ignore_errors=True)
        return jsonify({'error': f'保存文件失败: {str(e)}'}), 500

    # 创建任务队列和存储
    q = queue.Queue()
    with tasks_lock:
        tasks[task_id] = {
            'queue': q,
            'status': 'pending',
            'result': None,
            'error': None
        }

    # 启动后台处理线程，传递文件路径列表和临时目录
    thread = threading.Thread(
        target=process_task,
        args=(task_id, saved_paths, temp_upload_dir, age, gender, ai_model)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'task_id': task_id, 'status': 'pending'})

@app.route('/api/progress/<task_id>')
def progress_stream(task_id):
    """SSE 流式推送任务进度"""
    with tasks_lock:
        task = tasks.get(task_id)
        if not task:
            return jsonify({'error': '任务不存在'}), 404
        q = task['queue']

    def generate():
        while True:
            try:
                msg = q.get(timeout=1)
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                if msg.get('step') in ('done', 'error'):
                    break
            except queue.Empty:
                with tasks_lock:
                    status = tasks.get(task_id, {}).get('status')
                if status in ('done', 'error'):
                    if status == 'done':
                        yield f"data: {json.dumps({'step': 'done', 'message': '分析完成', 'status': 'complete'})}\n\n"
                    else:
                        err_msg = tasks.get(task_id, {}).get('error', '未知错误')
                        yield f"data: {json.dumps({'step': 'error', 'message': err_msg, 'status': 'error'})}\n\n"
                    break
                continue

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/result/<task_id>', methods=['GET'])
def get_result(task_id):
    """获取任务最终结果"""
    with tasks_lock:
        task = tasks.get(task_id)
        if not task:
            return jsonify({'error': '任务不存在'}), 404
        if task['status'] == 'done' and task['result']:
            return jsonify(task['result'])
        elif task['status'] == 'error':
            return jsonify({'error': task['error']}), 500
        else:
            return jsonify({'status': task['status'], 'message': '任务尚未完成'}), 202

# ================ 原有API路由（保留兼容性） ================
@app.route('/api/predict', methods=['POST'])
def predict():
    """同步预测接口（已弃用）"""
    return jsonify({'error': '此接口已弃用，请使用 /api/start_predict + SSE 获取进度'}), 410

@app.route('/')
def index():
    return jsonify({
        'name': 'AD辅助诊断API - 异步进度版',
        'version': '3.1.0',
        'status': '运行中',
        'models_loaded': {
            'random_forest': rf_model is not None
        },
        'feature_extractor_ready': feature_extractor is not None,
        'features_count': len(selected_features) if selected_features else 0,
        'model_accuracy': model_accuracy,
        'endpoints': {
            'async_start': '/api/start_predict (POST)',
            'progress': '/api/progress/<task_id> (GET, SSE)',
            'result': '/api/result/<task_id> (GET)',
            'health': '/api/health (GET)',
            'models': '/api/models (GET)'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'AD辅助诊断API运行正常',
        'timestamp': datetime.datetime.now().isoformat(),
        'models_available': {
            'random_forest': rf_model is not None
        }
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'status': 'success',
        'models': {
            'random_forest': {
                'name': '随机森林',
                'loaded': rf_model is not None,
                'accuracy': model_accuracy,
                'description': '基于多棵决策树的集成学习算法，对异常值和噪声具有较好的鲁棒性'
            }
        },
        'features_count': len(selected_features) if selected_features else 0
    })

@app.route('/api/test/upload', methods=['POST'])
def test_upload():
    """测试上传功能"""
    if 'image' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    temp_dir = tempfile.mkdtemp(dir=UPLOAD_FOLDER)
    filepath = os.path.join(temp_dir, secure_filename(file.filename))
    file.save(filepath)
    
    file_info = {
        'saved_path': filepath,
        'file_size': os.path.getsize(filepath),
        'file_exists': os.path.exists(filepath)
    }
    
    os.remove(filepath)
    os.rmdir(temp_dir)
    
    return jsonify({
        'status': 'success',
        'message': '文件上传测试成功',
        'file_info': file_info
    })

# ================ 启动服务器 ================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("AD辅助诊断系统 - 异步进度版")
    print("="*60)
    print(f"工作目录: {os.getcwd()}")
    print(f"上传目录: {UPLOAD_FOLDER}")
    print(f"随机森林模型: {'✅' if rf_model else '❌'}")
    print(f"特征提取器: {'✅' if feature_extractor else '❌'}")
    print(f"特征数量: {len(selected_features)}")
    print("\n可用端点:")
    print("  GET  /                         - 首页")
    print("  GET  /api/health                - 健康检查")
    print("  GET  /api/models                 - 获取模型信息")
    print("  POST /api/start_predict          - 启动异步预测")
    print("  GET  /api/progress/<task_id>     - SSE进度流")
    print("  GET  /api/result/<task_id>       - 获取最终结果")
    print("  POST /api/test/upload             - 测试上传")
    print("\n启动服务器: http://localhost:5000")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)