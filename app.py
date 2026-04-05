import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import xgboost as xgb
from model_logic import get_model_insights

# Set page configuration
st.set_page_config(
    page_title='🛡️ UPI Loan Trust Gate',
    page_icon='🛡️',
    layout='wide'
)

# Main header
st.title('🛡️ UPI Loan Trust Gate')

# Load model artifacts from local directory
@st.cache_resource
def load_model_artifacts():
    """Load the XGBoost model, encoders, and metadata from local app directory"""
    # Get the directory where this script is located
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Load XGBoost model from native JSON format
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(base_path, 'upi_model.json'))
        
        # Load encoders and metadata using joblib
        encoders = joblib.load(os.path.join(base_path, 'encoders.pkl'))
        metadata = joblib.load(os.path.join(base_path, 'metadata.pkl'))
        
        return model, encoders, metadata, None
    except Exception as e:
        return None, None, None, str(e)

# Load model artifacts
with st.spinner('🔄 Loading model...'):
    model, encoders, metadata, error = load_model_artifacts()

if error:
    st.error(f'❌ Error loading model artifacts: {error}')
    st.info('Please ensure the following files exist in the app directory:\n- upi_model.json\n- encoders.pkl\n- metadata.pkl')
    st.stop()

st.success('✅ Model loaded successfully!')

# File uploader
st.subheader('Upload your last 50 UPI transactions (CSV)')
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file containing your last 50 UPI transactions"
)

# Process uploaded file
if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        st.write(f"📊 Loaded {len(df)} transactions")
        
        # Show preview of data
        with st.expander("Preview uploaded data"):
            st.dataframe(df.head(10))
        
        # Add analysis button
        if st.button('🔍 Analyze Trust Score', type='primary'):
            with st.spinner('🔄 Analyzing your transaction history...'):
                # Call get_model_insights
                trust_score, reasons = get_model_insights(df, model, encoders, metadata)
            
            # Display results
            st.markdown("---")
            st.subheader('📊 Results Dashboard')
            
            # Create two columns for dashboard
            col1, col2 = st.columns([1, 1])
            
            # COLUMN 1: PLOTLY GAUGE CHART
            with col1:
                # Determine color based on score
                if trust_score >= 80:
                    gauge_color = "green"
                elif trust_score >= 60:
                    gauge_color = "yellow"
                else:
                    gauge_color = "red"
                
                # Create Plotly Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = trust_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Trust Score", 'font': {'size': 24}},
                    delta = {'reference': 75, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                        'bar': {'color': gauge_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 60], 'color': 'rgba(255, 0, 0, 0.2)'},
                            {'range': [60, 80], 'color': 'rgba(255, 255, 0, 0.2)'},
                            {'range': [80, 100], 'color': 'rgba(0, 255, 0, 0.2)'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 75
                        }
                    }
                ))
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=60, b=20),
                    paper_bgcolor="white",
                    font={'color': "darkgray", 'family': "Arial"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show metric comparison to standard user
                standard_avg = 75
                score_delta = trust_score - standard_avg
                st.metric(
                    label="Comparison to Standard User",
                    value=f"{trust_score:.1f}",
                    delta=f"{score_delta:+.1f} vs avg (75.0)"
                )
            
            # COLUMN 2: FINAL VERDICT CARD
            with col2:
                st.markdown("### Final Verdict")
                
                # Determine verdict based on score
                if trust_score >= 80:
                    # Green background - APPROVED
                    verdict_html = """
                    <div style='background-color: #d4edda; border: 2px solid #28a745; border-radius: 10px; padding: 30px; text-align: center;'>
                        <h1 style='color: #155724; margin: 0;'>✅ LOAN PRE-APPROVED</h1>
                        <p style='color: #155724; font-size: 18px; margin-top: 15px;'>
                            Your transaction profile demonstrates excellent trust signals. 
                            You are pre-qualified for instant loan approval.
                        </p>
                    </div>
                    """
                    st.markdown(verdict_html, unsafe_allow_html=True)
                    
                elif trust_score >= 60:
                    # Yellow background - MANUAL VERIFICATION
                    verdict_html = """
                    <div style='background-color: #fff3cd; border: 2px solid #ffc107; border-radius: 10px; padding: 30px; text-align: center;'>
                        <h1 style='color: #856404; margin: 0;'>⚠️ MANUAL VERIFICATION REQUIRED</h1>
                        <p style='color: #856404; font-size: 18px; margin-top: 15px;'>
                            Your transaction profile requires additional review. 
                            Please submit supporting documents for manual verification.
                        </p>
                    </div>
                    """
                    st.markdown(verdict_html, unsafe_allow_html=True)
                    
                else:
                    # Red background - REJECTED
                    verdict_html = """
                    <div style='background-color: #f8d7da; border: 2px solid #dc3545; border-radius: 10px; padding: 30px; text-align: center;'>
                        <h1 style='color: #721c24; margin: 0;'>❌ LOAN APPLICATION REJECTED</h1>
                        <p style='color: #721c24; font-size: 18px; margin-top: 15px;'>
                            Your transaction profile indicates high risk. 
                            Please review the concerns below and reapply after improvements.
                        </p>
                    </div>
                    """
                    st.markdown(verdict_html, unsafe_allow_html=True)
            
            # WHY THIS SCORE? - DETAILED RISK ANALYSIS SECTION
            st.markdown("---")
            
            with st.expander("🔍 View Detailed Risk Analysis", expanded=True):
                st.markdown("### Why this score?")
                st.markdown("Our XGBoost model analyzed your transaction patterns and identified the following key factors:")
                
                if reasons:
                    for idx, reason in enumerate(reasons, 1):
                        reason_type = reason.get('type', 'neutral')
                        feature = reason.get('feature', 'Unknown')
                        text = reason.get('text', 'No details available')
                        
                        if reason_type == 'positive':
                            # Green success message for positive factors
                            st.success(f"✅ **{feature.title()}**: {text}")
                        elif reason_type == 'negative':
                            # Red error message for negative factors
                            st.error(f"⚠️ **{feature.title()}**: {text}")
                        else:
                            # Neutral info message
                            st.info(f"ℹ️ **{feature.title()}**: {text}")
                else:
                    st.info("No specific risk factors identified in your transaction history.")
                
                st.markdown("---")
                st.markdown("""
                    **How we calculate your Trust Score:**
                    
                    1. **Transaction Pattern Analysis**: We examine your spending habits, frequency, and amounts
                    2. **Risk Indicators**: We identify anomalies or suspicious behaviors in your payment history
                    3. **SHAP Explainability**: We use machine learning to determine which factors most influenced your score
                    4. **Soft-Maximum Aggregation**: We calculate an overall risk score from individual transaction risks
                """)
            
            # DATA PRIVACY FOOTER
            st.markdown("---")
            st.markdown("""
                <div style='background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 15px; border-radius: 5px; margin-top: 20px;'>
                    <h4 style='margin: 0 0 10px 0; color: #495057;'>🔒 Data Privacy & Security</h4>
                    <p style='margin: 0; color: #6c757d; font-size: 14px;'>
                        Your transaction data is processed locally within Databricks' secure environment. 
                        The XGBoost model analyzes your data in real-time without storing or transmitting 
                        any personal information externally. All computations are performed on encrypted infrastructure, 
                        and your uploaded CSV file is never saved to disk. Your privacy and data security are our top priorities.
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f'❌ Error processing file: {str(e)}')
        st.info('Please ensure your CSV file has the correct format and required columns.')

else:
    st.info('👆 Please upload a CSV file to begin analysis')
