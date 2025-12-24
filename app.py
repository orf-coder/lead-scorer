import streamlit as st
import pandas as pd
import logging
import os
import subprocess
from datetime import datetime
from lead_scorer import (
    judge_lead,
    load_all_saved_models,
    _get_prediction_and_probs,
    rule_score_to_confidence,
    save_lead_to_csv,
)

def parse_lead_from_text(text):
    lines = text.split('\n')
    data = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            data[key] = value
    return data

# Configure logging
log_file = 'lead_scorer_app.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Theme state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Scoring state
if 'scored' not in st.session_state:
    st.session_state.scored = False

# Page configuration
st.set_page_config(
    page_title="Lead Scorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme toggle at top
col_toggle, col_title = st.columns([1, 10])

with col_toggle:
    if st.button("☀️" if st.session_state.theme == 'light' else "🌙"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

with col_title:
    st.title("📊 Lead Scoring System")
    st.markdown("Hybrid rule-based and machine learning lead classification")

# Apply theme
light_css = """
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.hot-label { color: #ff4444; font-weight: bold; }
.warm-label { color: #ff8800; font-weight: bold; }
.cold-label { color: #0088ff; font-weight: bold; }
[data-testid="stAppViewContainer"] {
    background-color: #e0e0e0;
    color: black;
}
[data-testid="stSidebar"] {
    background-color: #e0e0e0;
    color: black;
}
.stSidebar * {
    color: black !important;
}
h1, h2, h3, h4, h5, h6 {
    color: black !important;
    font-weight: bold !important;
}
.stTextInput label, .stSelectbox label, .stTextArea label, .stFileUploader label, .stCheckbox label, .stSegmentedControl label {
    color: black !important;
}
.stSegmentedControl div {
    background-color: black !important;
    color: white !important;
}
</style>
"""

dark_css = """
<style>
.metric-card {
    background-color: #2e2e2e;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    color: white;
}
.hot-label { color: #ff6666; font-weight: bold; }
.warm-label { color: #ffaa00; font-weight: bold; }
.cold-label { color: #66aaff; font-weight: bold; }
[data-testid="stAppViewContainer"] {
    background-color: black;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: black;
    color: white;
}
.stSidebar * {
    color: white !important;
}
h1, h2, h3, h4, h5, h6 {
    color: white !important;
    font-weight: bold !important;
}
.stTextInput label, .stSelectbox label, .stTextArea label, .stFileUploader label, .stCheckbox label {
    color: white !important;
}
</style>
"""

if st.session_state.theme == 'light':
    st.markdown(light_css, unsafe_allow_html=True)
else:
    st.markdown(dark_css, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system combines:
    - **Rule-based scoring** (keywords, job title, source)
    - **ML models** (Logistic Regression, SVM, Naive Bayes)
    - **Confidence comparison** (rule vs ML models)
    """)
    st.divider()
    st.markdown("**Models Available:**")
    st.text("• Logistic Regression\n• Support Vector Machine\n• Naive Bayes\n• Ensemble")

# Main form
st.header("Enter Lead Information")

input_method = st.radio("Choose input method:", ["Manual Entry", "Upload TXT File"], key="input_method")

if input_method == "Manual Entry":
    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Name", placeholder="John Doe", key="name")
        company = st.text_input("Company", placeholder="Acme Corp", key="company")

    with col2:
        job_title = st.text_input("Job Title", placeholder="CTO, Director, etc.", key="job_title")
        source = st.segmented_control(
            "Source",
            ["email", "LinkedIn", "contact form", "website form", "phone"],
            key="source"
        )

    message = st.text_area(
        "Message / Lead Description",
        placeholder="Enter the lead message or description here...",
        height=120,
        key="message"
    )
else:
    uploaded_file = st.file_uploader("Upload a TXT file containing the lead information", type=['txt'], key="uploaded_file")

    if uploaded_file is not None:
        text = uploaded_file.read().decode('utf-8')
        parsed = parse_lead_from_text(text)
        st.session_state.parsed_name = parsed.get('name', '')
        st.session_state.parsed_company = parsed.get('company', '')
        st.session_state.parsed_job_title = parsed.get('job_title', '')
        st.session_state.parsed_source = parsed.get('source', '')
        st.session_state.message_content = parsed.get('message', text)
        st.text_area("Lead information from file:", value=text, height=120, disabled=True, key="message_display")
    else:
        st.session_state.message_content = ""
        st.warning("Please upload a TXT file.")

# Scoring button
if st.button("🔍 Score Lead", type="primary", use_container_width=True):
    if input_method == "Manual Entry":
        name = st.session_state.name
        company = st.session_state.company
        job_title = st.session_state.job_title
        source = st.session_state.source
        final_message = st.session_state.message
    else:
        name = st.session_state.parsed_name
        company = st.session_state.parsed_company
        job_title = st.session_state.parsed_job_title
        source = st.session_state.parsed_source
        final_message = st.session_state.message_content

    logging.info(f"Lead scoring initiated - Name: {name}, Company: {company}, Job: {job_title}, Source: {source}, Message length: {len(final_message)}")
    if not final_message.strip():
        st.error("Please enter a message to score")
        logging.warning("Scoring attempted without message")
        st.session_state.scored = False
    else:
        # Compute rule-based score
        rule_score, rule_label, rule_reason, _ = judge_lead(final_message, job_title, source, company)
        rule_confidence = rule_score_to_confidence(rule_score)
        logging.info(f"Rule-based scoring - Score: {rule_score}, Label: {rule_label}, Confidence: {rule_confidence:.2%}")
        final_label = rule_label

        # Initialize variables
        consensus = rule_label
        ml_preds = []

        # Load and predict with all ML models
        combined_input = f"{final_message} {job_title} {company}".strip()
        models_loaded = load_all_saved_models()
        ml_results = []
        for m in models_loaded:
            try:
                pred, top_prob, probs_dict = _get_prediction_and_probs(m['pipeline'], combined_input)
                ml_results.append({
                    'name': m.get('name', 'model'),
                    'pred': pred,
                    'top_prob': top_prob,
                    'probs': probs_dict
                })
                top_str = f"{top_prob:.2%}" if top_prob is not None else "N/A"
                logging.info(f"ML Model {m.get('name')}: Prediction {pred}, Confidence {top_str}")
            except Exception as e:
                logging.error(f"Failed to load/predict with model {m.get('name')}: {e}")
                st.warning(f"Could not load model {m.get('name')}: {e}")

        # Store results in session state
        st.session_state.scored = True
        st.session_state.rule_score = rule_score
        st.session_state.rule_label = rule_label
        st.session_state.rule_reason = rule_reason
        st.session_state.rule_confidence = rule_confidence
        st.session_state.ml_results = ml_results
        st.session_state.consensus = consensus
        st.session_state.final_label = final_label
        st.session_state.final_message = final_message
        st.session_state.scored_name = name
        st.session_state.scored_company = company
        st.session_state.scored_job_title = job_title
        st.session_state.scored_source = source

# Display results if scored
if st.session_state.scored:
    rule_score = st.session_state.rule_score
    rule_label = st.session_state.rule_label
    rule_reason = st.session_state.rule_reason
    rule_confidence = st.session_state.rule_confidence
    ml_results = st.session_state.ml_results
    consensus = st.session_state.consensus
    final_label = st.session_state.final_label
    final_message = st.session_state.final_message
    name = st.session_state.scored_name
    company = st.session_state.scored_company
    job_title = st.session_state.scored_job_title
    source = st.session_state.scored_source

    # Display rule-based result
    st.divider()
    st.header("📋 Rule-Based Scoring")

    rule_col1, rule_col2 = st.columns(2)

    with rule_col1:
        st.metric("Rule Score", f"{rule_score}", delta="Integer Score")

    with rule_col2:
        label_class = f"{rule_label}-label"
        st.markdown(f"<div class='{label_class}'>Label: {rule_label}</div>", unsafe_allow_html=True)

    st.info(f"**Reason**: {rule_reason}")

    # Display ML results
    if ml_results:
        st.divider()
        st.header("🤖 Machine Learning Models")

        # Create ML results dataframe for display
        ml_data = []
        for r in ml_results:
            ml_data.append({
                "Model": r['name'],
                "Prediction": r['pred']
            })

        ml_df = pd.DataFrame(ml_data)
        st.dataframe(ml_df, use_container_width=True, hide_index=True)


        # Comparison: Rule vs ML
        st.divider()
        st.header("⚖️ Rule vs ML Comparison")

        comparison_data = []
        for r in ml_results:
            comparison_data.append({
                "Model": r['name'],
                "ML Prediction": r['pred'],
                "Rule Prediction": rule_label,
                "Match": "✅" if r['pred'] == rule_label else "❌"
            })

        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Summary and recommendation
        st.divider()
        st.header("📌 Summary & Recommendation")

        # Determine consensus
        ml_preds = [r['pred'] for r in ml_results if r['pred'] is not None]
        ml_confs = [r['top_prob'] for r in ml_results if r['top_prob'] is not None]
        if ml_preds:
            from collections import Counter
            pred_counts = Counter(ml_preds)
            most_common_ml = pred_counts.most_common(1)[0][0]
            consensus = most_common_ml
            average_ml_conf = sum(ml_confs) / len(ml_confs) if ml_confs else None
        else:
            consensus = rule_label
            average_ml_conf = None

        # Calculate final confidence as average of rule and ML
        if average_ml_conf is not None:
            final_confidence = (rule_confidence + average_ml_conf) / 2
        else:
            final_confidence = rule_confidence

        final_label = consensus

        # Override final label based on final confidence thresholds
        if final_confidence < 0.5:
            final_label = 'Cold'
        elif final_confidence > 0.75:
            final_label = 'Hot'

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rule Prediction", rule_label)

        with col2:
            if ml_preds:
                st.metric("ML Consensus", consensus)
            else:
                st.metric("ML Consensus", "N/A")

        with col3:
            st.markdown(f"<h3 style='text-align: center;'>Final: <span class='{final_label.lower()}-label'>{final_label}</span></h3>",
                        unsafe_allow_html=True)

        with col4:
            st.metric("Final Confidence", f"{final_confidence:.1%}")

        # Recommendation based on final label
        st.divider()
        st.subheader("🎯 Recommended Action")
        if final_label == 'Hot':
            options = ["Assign to SDR", "Schedule Discovery Call", "Send Premium Email Template", "Other"]
        elif final_label == 'Warm':
            options = ["Send Follow-up Email", "Add to Nurture Campaign", "Offer Free Trial", "Other"]
        else:  # Cold
            options = ["Discard Lead", "Add to Cold List", "Monitor for Future Engagement", "Other"]

        selected_action = st.radio("Select action to take:", options, index=0, key="action_radio")
        if selected_action == "Other":
            other_action = st.text_input("Specify other action:", key="other_action")
            final_action = other_action if other_action else "Other (not specified)"
        else:
            final_action = selected_action
        st.info(f"Selected Action: **{final_action}**")

        # Export single result
        st.divider()
        single_result = {
            'Name': name,
            'Company': company,
            'Job_Title': job_title,
            'Source': source,
            'Message': final_message,
            'Rule_Label': rule_label,
            'ML_Consensus': consensus if ml_preds else 'N/A',
            'Final_Label': final_label
        }
        if st.button("💾 Save Lead to Training Data", use_container_width=True):
            # Use absolute path based on script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, "Data", "csvfile.csv")
            success = save_lead_to_csv(name, company, job_title, source, final_message, final_label, csv_path=csv_path)
            if success:
                st.success(f"✅ Lead saved as **{final_label}** to Data/csvfile.csv for training")
                logging.info(f"Lead saved to training data - Label: {final_label}")
                # Trigger retraining check in background
                subprocess.Popen(["python", "auto_retrain.py"])
                st.info("Entry saved successfully to CSV.")
            else:
                st.error("Failed to save lead to CSV after multiple attempts. Please ensure the file is not open in another program.")
                logging.error("Failed to save lead to CSV")

    else:
        st.warning("Error! No ML models found. Please run `python run_train_all.py` to train models.")
        logging.warning("Scoring attempted but no ML models loaded")

st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
Lead Scoring System v1.0 | Built with Streamlit<br>
Rule-Based + ML Hybrid Approach
</div>
""", unsafe_allow_html=True)
