import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import io
from datetime import datetime

# Utility functions
def hash_customer_id(customer_id):
    return hashlib.sha256(str(customer_id).encode()).hexdigest()

def calculate_rfm_score(row):
    return int(f"{row['R_Score']}{row['F_Score']}{row['M_Score']}")

def assign_customer_segment(score):
    if score >= 555:
        return "Champions"
    elif 445 <= score <= 554:
        return "Loyal Customers"
    elif 334 <= score <= 444:
        return "Potential Loyalists"
    elif 511 <= score <= 555:
        return "New Customers"
    elif 412 <= score <= 444:
        return "Promising"
    elif 322 <= score <= 411:
        return "Need Attention"
    elif 311 <= score <= 322:
        return "About to Sleep"
    elif 222 <= score <= 310:
        return "At Risk"
    elif 145 <= score <= 311:
        return "Can't Lose Them"
    else:
        return "Hibernating"

def get_segment_recommendations(segment):
    recommendations = {
        "Champions": [
            "Provide early access to new products or exclusive offers",
            "Give enhanced benefits in your loyalty program",
            "Encourage referrals with rewards"
        ],
        "Loyal Customers": [
            "Send personalized product suggestions",
            "Use targeted email campaigns for upselling and cross-selling",
            "Offer rewards for reviews or social media engagement"
        ],
        "Potential Loyalists": [
            "Offer time-sensitive discounts or free shipping on next purchase",
            "Send free samples of new or premium products",
            "Ask for feedback or reviews on recent purchases"
        ],
        "New Customers": [
            "Set up automated email sequences for onboarding",
            "Offer a discount for their second purchase",
            "Provide educational content about your products or brand"
        ],
        "Promising": [
            "Send personalized offers based on browsing behavior",
            "Send follow-up emails after purchase",
            "Offer discounts on abandoned carts"
        ],
        "Need Attention": [
            "Send reactivation emails with special discounts",
            "Show reviews, testimonials, or social proof",
            "Use limited-time promotions to encourage purchases"
        ],
        "About to Sleep": [
            "Send a 'We miss you' email with an aggressive discount",
            "Offer free shipping as an incentive",
            "Send a reactivation survey"
        ],
        "At Risk": [
            "Send targeted emails highlighting changes or new features",
            "Conduct engagement surveys for feedback",
            "Offer strong incentives like 30-50% off next purchase"
        ],
        "Can't Lose Them": [
            "Reach out with urgent re-engagement campaigns",
            "Provide exclusive VIP offers",
            "Conduct direct surveys or calls to understand needs"
        ],
        "Hibernating": [
            "Send a final 'Come back' email with a significant discount",
            "Consider removing from frequent marketing emails",
            "Run tests with different content types for re-engagement"
        ]
    }
    return recommendations.get(segment, ["No specific recommendations available."])

def fetch_shopify_data(api_key, shop_url):
    # Placeholder function - replace with actual Shopify API integration
    st.warning("This is a placeholder for Shopify integration. In a real implementation, this would fetch data from your Shopify store.")
    return pd.DataFrame()  # Return empty DataFrame for now

def fetch_mailchimp_data(api_key, list_id):
    # Placeholder function - replace with actual Mailchimp API integration
    st.warning("This is a placeholder for Mailchimp integration. In a real implementation, this would fetch data from your Mailchimp account.")
    return pd.DataFrame()  # Return empty DataFrame for now

def calculate_customer_lifetime_value(df):
    return df.groupby('hashed_customer_id')['amount'].sum()

def predict_churn_probability(rfm):
    # This is a simplified placeholder. In a real scenario, you'd use a more sophisticated model.
    return 1 - (rfm['RFM_Score'] / 555)

def generate_personalized_campaigns(segment):
    campaigns = {
        "Champions": ["VIP exclusive product launch", "Brand ambassador program"],
        "Loyal Customers": ["Loyalty program tier upgrade", "Personalized product bundle"],
        "Potential Loyalists": ["Limited time offer on favorite categories", "Early access to sales"],
        "New Customers": ["Welcome series with product education", "First purchase anniversary offer"],
        "Promising": ["Category-specific discounts", "Engagement-driven rewards"],
        "Need Attention": ["Reactivation series with incremental offers", "Feedback survey with incentive"],
        "About to Sleep": ["Last chance offer", "Product recommendation quiz"],
        "At Risk": ["Win-back campaign with high-value incentive", "Personal outreach from customer service"],
        "Can't Lose Them": ["VIP concierge service", "Exclusive offline event invitation"],
        "Hibernating": ["Reintroduction to new products/features", "Surprise and delight campaign"]
    }
    return campaigns.get(segment, ["No specific campaigns available."])

# Streamlit app
st.set_page_config(page_title="E-commerce RFM Analytics Suite", page_icon="📊", layout="wide")

st.title('🚀 E-commerce RFM Analytics Suite')
st.markdown("""
This powerful analytics tool is designed exclusively for e-commerce businesses with 11-50 employees. 
Harness the power of RFM (Recency, Frequency, Monetary) analysis to supercharge your marketing strategy and skyrocket your revenue!
""")

# Data Source Selection
data_source = st.radio(
    "Choose your data source:",
    ('Upload CSV', 'Connect Shopify', 'Connect Mailchimp')
)

if data_source == 'Upload CSV':
    uploaded_file = st.file_uploader("Upload your customer data CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
elif data_source == 'Connect Shopify':
    st.info("Shopify integration coming soon! For now, please use the CSV upload option.")
    api_key = st.text_input("Enter your Shopify API key")
    shop_url = st.text_input("Enter your Shopify store URL")
    if st.button("Fetch Shopify Data"):
        df = fetch_shopify_data(api_key, shop_url)
elif data_source == 'Connect Mailchimp':
    st.info("Mailchimp integration coming soon! For now, please use the CSV upload option.")
    api_key = st.text_input("Enter your Mailchimp API key")
    list_id = st.text_input("Enter your Mailchimp list ID")
    if st.button("Fetch Mailchimp Data"):
        df = fetch_mailchimp_data(api_key, list_id)

if 'df' in locals() and not df.empty:
    # Check if required columns are present
    required_columns = ['customer_id', 'order_date', 'amount']
    if not all(column in df.columns for column in required_columns):
        st.error(f"The CSV file must contain the columns: {', '.join(required_columns)}")
    else:
        # Data Preprocessing
        df['hashed_customer_id'] = df['customer_id'].apply(hash_customer_id)
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # RFM Analysis
        current_date = datetime.now()
        rfm = df.groupby('hashed_customer_id').agg({
            'order_date': lambda x: (current_date - x.max()).days,
            'customer_id': 'count',
            'amount': 'sum'
        })
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # RFM Scoring
        r_labels = range(5, 0, -1)
        f_labels = range(1, 6)
        m_labels = range(1, 6)
        r_quartiles = pd.qcut(rfm['Recency'], q=5, labels=r_labels)
        f_quartiles = pd.qcut(rfm['Frequency'], q=5, labels=f_labels)
        m_quartiles = pd.qcut(rfm['Monetary'], q=5, labels=m_labels)
        rfm['R_Score'] = r_quartiles
        rfm['F_Score'] = f_quartiles
        rfm['M_Score'] = m_quartiles
        rfm['RFM_Score'] = rfm.apply(calculate_rfm_score, axis=1)
        rfm['Customer_Segment'] = rfm['RFM_Score'].apply(assign_customer_segment)

        # Dashboard
        st.header("📊 RFM Analysis Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RFM Score Distribution")
            fig_rfm = px.histogram(rfm, x='RFM_Score', nbins=20, 
                                   labels={'RFM_Score': 'RFM Score', 'count': 'Number of Customers'},
                                   title='Distribution of RFM Scores')
            st.plotly_chart(fig_rfm)
        
        with col2:
            st.subheader("Customer Segment Distribution")
            segment_counts = rfm['Customer_Segment'].value_counts()
            fig_segments = px.pie(values=segment_counts.values, names=segment_counts.index, 
                                  title='Customer Segments')
            st.plotly_chart(fig_segments)
        
        st.subheader("RFM Segmentation Bubble Chart")
        fig_bubble = px.scatter(rfm, x='Recency', y='Frequency', size='Monetary', color='Customer_Segment',
                                hover_name=rfm.index, size_max=60,
                                labels={'Recency': 'Recency (days)', 'Frequency': 'Frequency (count)', 'Monetary': 'Monetary (total spend)'},
                                title='RFM Segmentation')
        st.plotly_chart(fig_bubble)

        # Advanced Analytics
        st.header("🧠 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Lifetime Value")
            clv = calculate_customer_lifetime_value(df)
            fig_clv = px.histogram(clv, nbins=20, labels={'value': 'Customer Lifetime Value'},
                                   title='Distribution of Customer Lifetime Value')
            st.plotly_chart(fig_clv)
        
        with col2:
            st.subheader("Churn Probability")
            churn_prob = predict_churn_probability(rfm)
            fig_churn = px.histogram(churn_prob, nbins=20, labels={'value': 'Churn Probability'},
                                     title='Distribution of Churn Probability')
            st.plotly_chart(fig_churn)

        # Actionable Insights
        st.header("🎯 Actionable Insights")
        selected_segment = st.selectbox("Choose a customer segment for personalized strategies:", 
                                        rfm['Customer_Segment'].unique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recommended Strategies")
            strategies = get_segment_recommendations(selected_segment)
            for strategy in strategies:
                st.write(f"• {strategy}")
        
        with col2:
            st.subheader("Personalized Campaign Ideas")
            campaigns = generate_personalized_campaigns(selected_segment)
            for campaign in campaigns:
                st.write(f"• {campaign}")

        # Next Steps
        st.header("💡 Next Steps to Boost Your Revenue")
        st.markdown("""
        1. **Implement Segmented Campaigns**: Use the personalized campaign ideas to create targeted marketing efforts.
        2. **Focus on High CLV Customers**: Identify and nurture relationships with your highest value customers.
        3. **Reduce Churn**: Develop retention strategies for segments with high churn probability.
        4. **Optimize Customer Journey**: Use RFM insights to improve your overall customer experience.
        5. **Regular Analysis**: Schedule regular RFM analysis to track the impact of your strategies over time.
        """)

        # Download options
        st.header("📥 Download Your Insights")
        
        # Excel report
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            rfm.to_excel(writer, sheet_name='RFM Analysis')
            pd.DataFrame(strategies, columns=['Strategies']).to_excel(writer, sheet_name='Recommended Strategies')
            pd.DataFrame(campaigns, columns=['Campaign Ideas']).to_excel(writer, sheet_name='Campaign Ideas')
        buffer.seek(0)
        st.download_button(
            label="Download Full RFM Report (Excel)",
            data=buffer,
            file_name="rfm_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # PDF report (placeholder)
        st.download_button(
            label="Download Executive Summary (PDF)",
            data=b"Placeholder for PDF report",  # Replace with actual PDF generation
            file_name="rfm_executive_summary.pdf",
            mime="application/pdf"
        )

        # Call-to-Action
        st.markdown("""
        ---
        ## 🚀 Ready to take your e-commerce marketing to the next level?
        This free tool is just the beginning. Imagine what we could do with a full integration of your data sources and custom-tailored strategies.
        
        ### Book a free consultation to learn how we can:
        - Fully integrate with your Shopify store and Mailchimp campaigns
        - Provide real-time RFM analysis and automated segmentation
        - Develop AI-driven predictive models for customer behavior
        - Create a custom dashboard tailored to your specific KPIs
        
        [Book Your Free Consultation Now](https://your-booking-link.com)
        """)

else:
    st.info("Upload your data or connect to a data source to get started!")

# Footer
st.markdown("""
---
Made with ❤️ by Your Company Name | [Terms of Service](https://your-tos-link.com) | [Privacy Policy](https://your-privacy-policy-link.com)
""")