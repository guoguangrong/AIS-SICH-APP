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

# 特征中文名称
feature_cn_names = {
    "bnp_total": "基线BNP", "sbp_baseline": "基线收缩压", "opt": "发病至穿刺时间",
    "nihss_admit": "入院NIHSS评分", "aptt_total": "基线APTT", "age": "年龄",
    "agitation": "躁动情况", "anc_total": "基线中性粒细胞计数", "af": "房颤病史"
}

# 风险阈值和OR值（基于论文）
risk_thresholds = {
    "age": {"threshold": 74, "unit": "岁", "or_value": 3.36, "weight": 2},
    "nihss_admit": {"threshold": 12, "unit": "分", "or_value": 2.45, "weight": 2},
    "sbp_baseline": {"threshold": 146, "unit": "mmHg", "or_value": 3.29, "weight": 2},
    "bnp_total": {"threshold": 1120, "unit": "pg/mL", "or_value": 3.49, "weight": 2},
    "aptt_total": {"threshold": 38.4, "unit": "秒", "or_value": 3.26, "weight": 2},
    "anc_total": {"threshold": 6.34, "unit": "×10^9/L", "or_value": 2.11, "weight": 1},
    "af": {"threshold": 1, "unit": "", "or_value": 14.08, "weight": 3},
    "agitation": {"levels": {0: 0, 1: 2, 2: 3, 3: 4}, "or_values": {1: 4.01, 2: 16.75, 3: 79.02}, "weight": "dynamic"},
    "opt": {"thresholds": [(300, 1), (600, 2)], "unit": "分钟", "weight": "dynamic"}
}

agitation_map = {0: "无躁动", 1: "轻度躁动", 2: "中度躁动", 3: "重度躁动"}

# CSS样式
st.markdown("""
<style>
    .stNumberInput label, .stSelectbox label { font-size: 16px !important; font-weight: 500 !important; }
    .stNumberInput input { font-size: 18px !important; }
    .prediction-card { border-radius: 20px; padding: 30px; text-align: center; box-shadow: 0 10px 40px rgba(0,0,0,0.1); }
    .risk-low { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .risk-medium { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .risk-high { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .prediction-title { font-size: 24px; font-weight: bold; color: white; margin-bottom: 15px; }
    .prediction-prob { font-size: 48px; font-weight: bold; color: white; margin: 20px 0; }
    .prediction-level { font-size: 28px; font-weight: bold; color: white; margin-bottom: 20px; }
    .prediction-advice { font-size: 16px; color: white; background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin-top: 15px; }
    .risk-factor-card { border-radius: 12px; padding: 15px; margin-bottom: 10px; border-left: 4px solid; }
    .risk-factor-high { border-left-color: #eb3349; background: #fff5f5; }
    .risk-factor-mid { border-left-color: #f5576c; background: #fffaf0; }
    .risk-factor-low { border-left-color: #11998e; background: #f0fff4; }
    .risk-factor-title { font-weight: bold; font-size: 15px; margin-bottom: 5px; }
    .risk-factor-value { font-size: 13px; color: #6c757d; }
    hr { margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

st.title("急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器")
st.markdown("### 请填写以下信息，点击预测获取风险评估结果")
st.markdown("---")

# 输入区域
left_col, right_col = st.columns([1.2, 0.8])

with left_col:
    col1, col2 = st.columns(2)
    
    with col1:
        bnp_total_num = st.number_input("基线BNP (pg/mL)", min_value=0.0, max_value=50000.0, value=476.0, step=10.0, format="%.0f")
        sbp_baseline_num = st.number_input("基线收缩压 (mmHg)", min_value=0.0, max_value=300.0, value=130.0, step=1.0, format="%.0f")
        opt_num = st.number_input("发病至穿刺时间 (分钟)", min_value=0.0, max_value=1440.0, value=450.0, step=5.0, format="%.0f")
        nihss_admit_num = st.number_input("入院NIHSS评分 (分)", min_value=0.0, max_value=42.0, value=10.0, step=1.0, format="%.0f")
    
    with col2:
        aptt_total_num = st.number_input("基线APTT (秒)", min_value=0.0, max_value=120.0, value=36.9, step=1.0, format="%.1f")
        age_num = st.number_input("年龄 (岁)", min_value=0.0, max_value=120.0, value=65.0, step=1.0, format="%.0f")
        agitation = st.selectbox("术后躁动情况", options=[0, 1, 2, 3], format_func=lambda x: agitation_map[x])
        anc_total_num = st.number_input("基线中性粒细胞计数 (×10^9/L)", min_value=0.0, max_value=50.0, value=8.8, step=0.5, format="%.1f")
        af = st.selectbox("房颤病史", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
    
    st.markdown("---")
    predict_btn = st.button("预测", type="primary", use_container_width=True)

# 定义基于规则的评分函数
def calculate_rule_based_risk(values_dict):
    """基于论文阈值的风险评分"""
    score = 0
    max_score = 25
    
    # 年龄
    if values_dict['age'] > 74:
        score += 3
    
    # NIHSS评分
    if values_dict['nihss_admit'] > 12:
        score += 3
    
    # 收缩压
    if values_dict['sbp_baseline'] > 146:
        score += 3
    
    # BNP
    if values_dict['bnp_total'] > 1120:
        score += 3
    
    # APTT
    if values_dict['aptt_total'] > 38.4:
        score += 3
    
    # ANC
    if values_dict['anc_total'] > 6.34:
        score += 2
    
    # 房颤
    if values_dict['af'] == 1:
        score += 4
    
    # 躁动
    agitation_scores = {0: 0, 1: 2, 2: 3, 3: 4}
    score += agitation_scores.get(values_dict['agitation'], 0)
    
    # OPT
    if values_dict['opt'] > 600:
        score += 2
    elif values_dict['opt'] > 300:
        score += 1
    
    # 计算概率（基于训练集的18.2%基线风险）
    baseline_risk = 0.182
    risk_multiplier = 1 + (score / max_score) * 3  # 最大风险放大4倍
    risk_prob = min(0.85, baseline_risk * risk_multiplier)
    
    return risk_prob, score

with right_col:
    st.markdown("### 📊 预测结果")
    prediction_placeholder = st.empty()
    st.markdown("### 🔍 风险指标分析")
    risk_analysis_placeholder = st.empty()
    
    if predict_btn:
        feature_values = [bnp_total_num, sbp_baseline_num, opt_num, nihss_admit_num,
                         aptt_total_num, age_num, agitation, anc_total_num, af]
        input_df = pd.DataFrame([feature_values], columns=feature_names)
        
        # ========== 方法1：模型预测 ==========
        try:
            proba = model.predict_proba(input_df)[0]
            model_risk = proba[1]  # 高风险概率
            # 校准模型输出（基于诊断结果，SVM输出约31%，需要校准）
            model_risk_calibrated = model_risk * 0.65  # 校准系数
            model_risk_calibrated = max(0.05, min(model_risk_calibrated, 0.85))
        except Exception as e:
            st.error(f"模型预测失败: {e}")
            model_risk_calibrated = 0.18  # 默认基线风险
        
        # ========== 方法2：基于规则的评分 ==========
        values_dict = {
            "age": age_num, "nihss_admit": nihss_admit_num,
            "sbp_baseline": sbp_baseline_num, "opt": opt_num,
            "af": af, "agitation": agitation, "bnp_total": bnp_total_num,
            "aptt_total": aptt_total_num, "anc_total": anc_total_num
        }
        rule_risk, rule_score = calculate_rule_based_risk(values_dict)
        
        # ========== 混合预测：加权平均 ==========
        # 模型权重0.5，规则权重0.5
        final_risk = model_risk_calibrated * 0.5 + rule_risk * 0.5
        
        # 根据风险因素数量动态调整权重
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
        
        # 当高风险因素较多时，增加规则评分的权重
        if high_risk_count >= 4:
            final_risk = model_risk_calibrated * 0.3 + rule_risk * 0.7
        elif high_risk_count <= 1:
            final_risk = model_risk_calibrated * 0.7 + rule_risk * 0.3
        
        # 最终概率
        risk_prob = final_risk
        
        # 风险等级划分
        if risk_prob < 0.30:
            pred_class = "低风险"
            advice = f"混合模型预测风险概率为 {risk_prob:.1%}（模型:{model_risk_calibrated:.1%}，规则:{rule_risk:.1%}），属于低风险。建议定期随访。"
            risk_class = "risk-low"
        elif risk_prob < 0.70:
            pred_class = "中风险"
            advice = f"混合模型预测风险概率为 {risk_prob:.1%}（模型:{model_risk_calibrated:.1%}，规则:{rule_risk:.1%}），属于中风险。建议密切观察。"
            risk_class = "risk-medium"
        else:
            pred_class = "高风险"
            advice = f"混合模型预测风险概率为 {risk_prob:.1%}（模型:{model_risk_calibrated:.1%}，规则:{rule_risk:.1%}），属于高风险。建议立即就医。"
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
        
        # 显示混合预测详情
        st.markdown(f"""
        <div style="background: #f0f0f0; border-radius: 12px; padding: 12px; margin-top: 10px;">
            <div style="font-size: 13px; color: #2c3e50;">
                <strong>🔬 混合预测详情：</strong><br>
                📊 模型预测: {model_risk_calibrated:.1%} | 📋 规则评分: {rule_risk:.1%} (得分: {rule_score}/25)<br>
                ⚠️ 高风险因素数量: {high_risk_count} 个
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 风险指标分析
        feature_values_dict = {
            "age": age_num, "nihss_admit": nihss_admit_num, "sbp_baseline": sbp_baseline_num,
            "opt": opt_num, "af": af, "agitation": agitation, "bnp_total": bnp_total_num,
            "aptt_total": aptt_total_num, "anc_total": anc_total_num
        }
        
        risk_indicators_html = '<div style="max-height: 400px; overflow-y: auto;">'
        
        for feature, threshold_info in risk_thresholds.items():
            value = feature_values_dict.get(feature, 0)
            cn_name = feature_cn_names.get(feature, feature)
            
            is_high_risk = False
            risk_desc = ""
            
            if feature == "af":
                is_high_risk = (value == 1)
                risk_desc = "⚠️ 高风险因素" if is_high_risk else "✓ 正常"
                display_value = "是" if value == 1 else "否"
            elif feature == "agitation":
                is_high_risk = (value >= 1)
                or_values = threshold_info.get("or_values", {})
                if value == 0:
                    risk_desc = "✓ 正常"
                elif value == 1:
                    risk_desc = f"⚠️ 轻度风险 (OR={or_values.get(1, 4.01)})"
                elif value == 2:
                    risk_desc = f"⚠️ 中度风险 (OR={or_values.get(2, 16.75)})"
                else:
                    risk_desc = f"⚠️ 重度风险 (OR={or_values.get(3, 79.02)})"
                display_value = agitation_map.get(value, "未知")
            elif feature == "opt":
                if value < 300:
                    risk_desc = "✓ 时间较短"
                elif value < 600:
                    risk_desc = "⚠️ 时间中等"
                else:
                    risk_desc = "⚠️ 时间较长"
                display_value = f"{value:.0f} 分钟 ({value/60:.1f} 小时)"
            else:
                threshold = threshold_info["threshold"]
                unit = threshold_info.get("unit", "")
                is_high_risk = (value > threshold)
                or_value = threshold_info.get("or_value", "")
                risk_desc = f"⚠️ 超过阈值 ({threshold}{unit})，OR={or_value}" if is_high_risk else f"✓ 低于阈值 ({threshold}{unit})"
                if feature == "anc_total":
                    display_value = f"{value:.1f} {unit}"
                elif feature == "aptt_total":
                    display_value = f"{value:.1f} {unit}"
                else:
                    display_value = f"{value:.0f} {unit}".strip()
            
            card_class = "risk-factor-high" if is_high_risk else ("risk-factor-mid" if feature == "opt" and value > 300 else "risk-factor-low")
            status_icon = "🔴" if is_high_risk else ("🟡" if feature == "opt" and value > 300 else "🟢")
            detail_desc = threshold_info.get("description", "")
            
            risk_indicators_html += f"""
            <div class="risk-factor-card {card_class}">
                <div class="risk-factor-title">{status_icon} {cn_name}: {display_value}</div>
                <div class="risk-factor-value">状态: {risk_desc}<br><span style="color: #6c757d; font-size: 12px;">📋 {detail_desc}</span></div>
            </div>
            """
        
        risk_indicators_html += '</div>'
        risk_indicators_html += f"""
        <div style="background: #e8f4f8; border-radius: 12px; padding: 12px; margin-top: 10px;">
            <div style="font-size: 13px; color: #2c3e50;"><strong>💡 综合建议：</strong><br>{advice}</div>
        </div>
        """
        risk_analysis_placeholder.markdown(risk_indicators_html, unsafe_allow_html=True)
        
        # 输入摘要
        with st.expander("查看完整输入信息"):
            opt_hours = opt_num / 60
            input_summary = pd.DataFrame({
                "变量名称": ["年龄", "入院NIHSS评分", "基线收缩压", "发病至穿刺时间", "房颤病史",
                            "躁动情况", "基线BNP", "基线APTT", "基线ANC"],
                "输入值": [f"{age_num:.0f} 岁", f"{nihss_admit_num:.0f} 分", f"{sbp_baseline_num:.0f} mmHg",
                          f"{opt_num:.0f} 分钟 ({opt_hours:.1f} 小时)", "是" if af == 1 else "否",
                          agitation_map[agitation], f"{bnp_total_num:.0f} pg/mL",
                          f"{aptt_total_num:.1f} 秒", f"{anc_total_num:.1f} ×10^9/L"]
            })
            st.dataframe(input_summary, use_container_width=True, hide_index=True)
    
    else:
        prediction_placeholder.markdown("""
        <div style="background: #f8f9fa; border-radius: 20px; padding: 60px 30px; text-align: center; border: 2px dashed #dee2e6;">
            <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
            <div style="font-size: 18px; color: #6c757d;">填写左侧信息后点击"预测"按钮</div>
        </div>
        """, unsafe_allow_html=True)
        risk_analysis_placeholder.markdown("""
        <div style="background: #f8f9fa; border-radius: 20px; padding: 40px 20px; text-align: center; border: 2px dashed #dee2e6;">
            <div style="font-size: 36px; margin-bottom: 15px;">🔍</div>
            <div style="font-size: 16px; color: #6c757d;">预测后将显示各风险指标的分析</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("注：本预测结果仅供参考，不能替代专业医疗建议。")