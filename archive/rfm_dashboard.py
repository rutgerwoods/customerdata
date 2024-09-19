import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import io
from datetime import datetime

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

# Titel van de app
st.title('Enhanced RFM Analysis Dashboard')

# CSV-upload functie
uploaded_file = st.file_uploader("Upload je klantdata CSV-bestand", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
       
        # Controleer of de benodigde kolommen aanwezig zijn
        required_columns = ['customer_id', 'order_date', 'amount']
        if not all(column in df.columns for column in required_columns):
            st.error(f"Het CSV-bestand moet de kolommen {', '.join(required_columns)} bevatten.")
        else:
            # Anonimiseer klant-ID's
            df['hashed_customer_id'] = df['customer_id'].apply(hash_customer_id)
            df = df.drop(columns=['customer_id'])
           
            # Toon de geüploade data
            st.write("Geüploade Data:")
            st.dataframe(df)
           
            # Verander de 'order_date' kolom naar datetime
            df['order_date'] = pd.to_datetime(df['order_date'])
           
            # Voer de RFM-analyse uit
            current_date = datetime.now()
            rfm = df.groupby('hashed_customer_id').agg({
                'order_date': lambda x: (current_date - x.max()).days,
                'hashed_customer_id': 'count',
                'amount': 'sum'
            })
           
            rfm.columns = ['Recency', 'Frequency', 'Monetary']
           
            # Bereken RFM-scores
            rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])
            rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
            rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5])
           
            rfm['RFM_Score'] = rfm.apply(calculate_rfm_score, axis=1)
            rfm['Customer_Segment'] = rfm['RFM_Score'].apply(assign_customer_segment)
           
            # Toon RFM-analyse resultaten
            st.write("RFM Analyse Resultaten:")
            st.dataframe(rfm)
           
            # RFM Score Verdeling
            st.write("RFM Score Verdeling:")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=rfm, x='RFM_Score', bins=20, kde=True, ax=ax)
            ax.set_xlabel('RFM Score')
            ax.set_ylabel('Aantal Klanten')
            ax.set_title('Verdeling van RFM Scores')
            st.pyplot(fig)
           
            # Customer Segment Verdeling
            st.write("Customer Segment Verdeling:")
            fig, ax = plt.subplots(figsize=(12, 6))
            segment_counts = rfm['Customer_Segment'].value_counts()
            sns.barplot(x=segment_counts.index, y=segment_counts.values, ax=ax)
            ax.set_xlabel('Customer Segment')
            ax.set_ylabel('Aantal Klanten')
            ax.set_title('Aantal Klanten per Segment')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
           
            # Heatmap van RFM-factoren
            st.write("Heatmap van RFM-factoren:")
            corr_matrix = rfm[['Recency', 'Frequency', 'Monetary']].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlatie tussen RFM-factoren')
            st.pyplot(fig)
           
            # Segment-specifieke aanbevelingen
            st.write("Segment-specifieke Aanbevelingen:")
            selected_segment = st.selectbox("Kies een segment voor aanbevelingen:", rfm['Customer_Segment'].unique())
            recommendations = get_segment_recommendations(selected_segment)
            for rec in recommendations:
                st.write(f"- {rec}")
           
            # Download RFM Analyse Rapport
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                rfm.to_excel(writer, sheet_name='RFM Analysis', index=True)
            buf.seek(0)
            st.download_button(label="Download RFM Analyse Rapport", data=buf, file_name="rfm_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"Er is een fout opgetreden: {e}")