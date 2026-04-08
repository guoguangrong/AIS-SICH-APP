# 导入核心库
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器",
    layout="wide"
)

# 加载模型（使用相对最稳定的 SVM）
@st.cache_resource
def load_model():
    return joblib.load('svm_model.pkl')

try:
    model = load_model()
    st.success("✅ 模型加载成功")
except FileNotFoundError:
    st.error("❌ svm_model.pkl 文件未找到")
    st.stop()
except Exception as e:
    st.error(f"❌ 模型加载失败: {e}")
    st.stop()

# 特征名称
feature_names = [
    "bnp_total", "sbp_baseline", "opt", "nihss_admit",
    "aptt_total", "age", "agitation", "anc_total", "af"
]

agitation_map = {0: "无躁动", 1: "轻度躁动", 2: "中度躁动", 3: "重度躁动"}

# CSS样式
st.markdown("""
<style>
    .stNumberInput label, .stSelectbox label {
        font-size: 18px !important;
        font-weight: 500 !important;
    }
    .stNumberInput input {
        font-size: 20px !important;
    }
    .stSelectbox div[data-baseweb="select"] span {
        font-size: 18px !important;
    }
    .prediction-card {
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-top: 30px;
    }
    .risk-low { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .risk-medium { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .risk-high { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .prediction-title { font-size: 24px; font-weight: bold; color: white; margin-bottom: 15px; }
    .prediction-prob { font-size: 48px; font-weight: bold; color: white; margin: 20px 0; }
    .prediction-level { font-size: 28px; font-weight: bold; color: white; margin-bottom: 20px; }
    .prediction-advice { font-size: 16px; color: white; background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin-top: 15px; }
    hr { margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

st.title("急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器")
st.markdown("### 请填写以下信息，点击预测获取风险评估结果")
st.markdown("---")

# ========== 三列布局 ==========
col_left, col_middle, col_right = st.columns([1.2, 1, 1])

with col_left:
    bnp_total_num = st.number_input("基线BNP (pg/mL)", min_value=0.0, max_value=50000.0, value=476.0, step=10.0, format="%.0f")
    sbp_baseline_num = st.number_input("基线收缩压 (mmHg)", min_value=0.0, max_value=300.0, value=130.0, step=1.0, format="%.0f")
    opt_num = st.number_input("发病至穿刺时间 (分钟)", min_value=0.0, max_value=30000.0, value=450.0, step=5.0, format="%.0f")
    nihss_admit_num = st.number_input("入院NIHSS评分 (分)", min_value=0.0, max_value=42.0, value=10.0, step=1.0, format="%.0f")
    anc_total_num = st.number_input("基线中性粒细胞计数 (×10^9/L)", min_value=0.0, max_value=50.0, value=8.8, step=0.5, format="%.1f")

with col_middle:
    aptt_total_num = st.number_input("基线APTT (秒)", min_value=0.0, max_value=12000.0, value=36.9, step=1.0, format="%.1f")
    age_num = st.number_input("年龄 (岁)", min_value=0.0, max_value=220.0, value=65.0, step=1.0, format="%.0f")
    agitation = st.selectbox("术后躁动情况", options=[0, 1, 2, 3], format_func=lambda x: agitation_map[x])
    af = st.selectbox("房颤病史", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
    
    st.markdown("---")
    predict_btn = st.button("预测", type="primary", use_container_width=True)

with col_right:
    prediction_placeholder = st.empty()
    risk_analysis_placeholder = st.empty()

# 定义基于规则的评分函数
def calculate_rule_based_risk(values_dict):
    """基于论文阈值的风险评分"""
    score = 0
    max_score = 25
    
    if values_dict['age'] > 74:
        score += 12
    if values_dict['nihss_admit'] > 12:
        score += 19
    if values_dict['sbp_baseline'] > 146:
        score += 13
    if values_dict['bnp_total'] > 1120:
        score += 17
    if values_dict['aptt_total'] > 38.4:
        score += 8
    if values_dict['anc_total'] > 6.34:
        score += 8
    if values_dict['af'] == 1:
        score += 10
    
    agitation_scores = {0: 0, 1: 19, 2: 22, 3: 25}
    score += agitation_scores.get(values_dict['agitation'], 0)
    
    if values_dict['opt'] > 600:
        score += 12
    elif values_dict['opt'] > 300:
        score += 10
    
    baseline_risk = 0.182
    risk_multiplier = 1 + (score / max_score) * 3
    risk_prob = min(0.85, baseline_risk * risk_multiplier)
    
    return risk_prob, score

if predict_btn:
    feature_values = [bnp_total_num, sbp_baseline_num, opt_num, nihss_admit_num,
                     aptt_total_num, age_num, agitation, anc_total_num, af]
    input_df = pd.DataFrame([feature_values], columns=feature_names)
    
    # ========== 方法1：模型预测 ==========
    try:
        proba = model.predict_proba(input_df)[0]
        model_risk = proba[1]
        model_risk_calibrated = model_risk * 0.65
        model_risk_calibrated = max(0.05, min(model_risk_calibrated, 0.85))
    except Exception as e:
        st.error(f"模型预测失败: {e}")
        model_risk_calibrated = 0.18
    
    # ========== 方法2：基于规则的评分 ==========
    values_dict = {
        "age": age_num, "nihss_admit": nihss_admit_num,
        "sbp_baseline": sbp_baseline_num, "opt": opt_num,
        "af": af, "agitation": agitation, "bnp_total": bnp_total_num,
        "aptt_total": aptt_total_num, "anc_total": anc_total_num
    }
    rule_risk, rule_score = calculate_rule_based_risk(values_dict)
    
    # ========== 混合预测 ==========
    final_risk = model_risk_calibrated * 0.25 + rule_risk * 0.75
    
    high_risk_count = 0
    if age_num > 74: high_risk_count += 1
    if nihss_admit_num > 12: high_risk_count += 1
    if sbp_baseline_num > 146: high_risk_count += 1
    if bnp_total_num > 1120: high_risk_count += 1
    if aptt_total_num > 38.4: high_risk_count += 1
    if anc_total_num > 6.34: high_risk_count += 1
    if af == 1: high_risk_count += 1
    if agitation >= 1: high_risk_count += 1
    if opt_num > 600: high_risk_count += 1
    
    if high_risk_count >= 4:
        final_risk = model_risk_calibrated * 0.3 + rule_risk * 0.7
    elif high_risk_count <= 1:
        final_risk = model_risk_calibrated * 0.7 + rule_risk * 0.3
    
    risk_prob = final_risk
    
    # 风险等级划分
    if risk_prob < 0.30:
        pred_class = "低风险"
        advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于低风险。建议继续保持当前治疗方案，定期随访。"
        risk_class = "risk-low"
    elif risk_prob < 0.70:
        pred_class = "中风险"
        advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于中风险。建议密切观察，遵医嘱进行相关检查。"
        risk_class = "risk-medium"
    else:
        pred_class = "高风险"
        advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于高风险。建议立即就医，加强监测和预防措施。"
        risk_class = "risk-high"
    
    # 显示预测结果
    prediction_placeholder.markdown(f"""
    <div class="prediction-card {risk_class}">
        <div class="prediction-title">📈 风险评估结果</div>
        <div class="prediction-prob">{risk_prob:.1%}</div>
        <div class="prediction-level">{pred_class}</div>
        <div class="prediction-advice">💡 {advice}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== 简化的风险指标分析 - 使用st.columns ==========
    
    # 定义指标数据
    indicators = []
    
    # 年龄
    if age_num > 74:
        indicators.append(("🔴 年龄", f"{age_num:.0f} 岁"))
    else:
        indicators.append(("🟢 年龄", f"{age_num:.0f} 岁"))
    
    # NIHSS评分
    if nihss_admit_num > 12:
        indicators.append(("🔴 入院NIHSS评分", f"{nihss_admit_num:.0f} 分"))
    else:
        indicators.append(("🟢 入院NIHSS评分", f"{nihss_admit_num:.0f} 分"))
    
    # 收缩压
    if sbp_baseline_num > 146:
        indicators.append(("🔴 基线收缩压", f"{sbp_baseline_num:.0f} mmHg"))
    else:
        indicators.append(("🟢 基线收缩压", f"{sbp_baseline_num:.0f} mmHg"))
    
    # BNP
    if bnp_total_num > 1120:
        indicators.append(("🔴 基线BNP", f"{bnp_total_num:.0f} pg/mL"))
    else:
        indicators.append(("🟢 基线BNP", f"{bnp_total_num:.0f} pg/mL"))
    
    # APTT
    if aptt_total_num > 38.4:
        indicators.append(("🔴 基线APTT", f"{aptt_total_num:.1f} 秒"))
    else:
        indicators.append(("🟢 基线APTT", f"{aptt_total_num:.1f} 秒"))
    
    # ANC
    if anc_total_num > 6.34:
        indicators.append(("🔴 基线中性粒细胞计数", f"{anc_total_num:.1f} ×10^9/L"))
    else:
        indicators.append(("🟢 基线中性粒细胞计数", f"{anc_total_num:.1f} ×10^9/L"))
    
    # 房颤
    if af == 1:
        indicators.append(("🔴 房颤病史", "是"))
    else:
        indicators.append(("🟢 房颤病史", "否"))
    
    # 躁动
    if agitation == 0:
        indicators.append(("🟢 躁动情况", agitation_map[agitation]))
    elif agitation == 1:
        indicators.append(("🟡 躁动情况", agitation_map[agitation]))
    else:
        indicators.append(("🔴 躁动情况", agitation_map[agitation]))
    
    # OPT
    if opt_num <= 300:
        indicators.append(("🟢 发病至穿刺时间", f"{opt_num:.0f} 分钟"))
    elif opt_num <= 600:
        indicators.append(("🟡 发病至穿刺时间", f"{opt_num:.0f} 分钟"))
    else:
        indicators.append(("🔴 发病至穿刺时间", f"{opt_num:.0f} 分钟"))
    
    # 使用Streamlit原生组件显示
    with risk_analysis_placeholder.container():
        st.markdown("##### 风险指标")
        
        # 分成两列显示
        col1, col2 = st.columns(2)
        
        # 前5个放左边
        with col1:
            for i in range(5):
                if i < len(indicators):
                    st.write(f"{indicators[i][0]}: {indicators[i][1]}")
        
        # 后4个放右边
        with col2:
            for i in range(5, len(indicators)):
                st.write(f"{indicators[i][0]}: {indicators[i][1]}")
        
        st.markdown("---")
        st.info(f"💡 {advice}")

else:
    prediction_placeholder.markdown("""
    <div style="background: #f8f9fa; border-radius: 20px; padding: 40px 20px; text-align: center; border: 2px dashed #dee2e6;">
        <div style="font-size: 36px; margin-bottom: 15px;">📊</div>
        <div style="font-size: 16px; color: #6c757d;">填写左侧信息后点击"预测"按钮</div>
        <div style="font-size: 13px; color: #adb5bd; margin-top: 8px;">将在此处显示风险评估结果</div>
    </div>
    """, unsafe_allow_html=True)
    risk_analysis_placeholder.empty()

st.markdown("---")
st.caption("注：本预测结果仅供参考，不能替代专业医疗建议。")