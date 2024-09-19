import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import io
from datetime import datetime

# Utility functions
def hash_customer_id(customer_id):
    return hashlib.sha256(str(customer_id).encode()).hexdigest()

def calculate_rfm_score(row):
    try:
        return int(f"{int(row['R_Score'])}{int(row['F_Score'])}{int(row['M_Score'])}")
    except ValueError:
        st.error(f"Error calculating RFM score for row: {row}")
        return 111  # Default score

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
            "Bied vroege toegang tot nieuwe producten of exclusieve aanbiedingen",
            "Geef verbeterde voordelen in uw loyaliteitsprogramma",
            "Moedig verwijzingen aan met beloningen"
        ],
        "Loyal Customers": [
            "Stuur gepersonaliseerde productaanbevelingen",
            "Gebruik gerichte e-mailcampagnes voor upselling en cross-selling",
            "Bied beloningen voor recensies of sociale media-betrokkenheid"
        ],
        "Potential Loyalists": [
            "Bied tijdgebonden kortingen of gratis verzending bij volgende aankoop",
            "Stuur gratis monsters van nieuwe of premium producten",
            "Vraag om feedback of recensies over recente aankopen"
        ],
        "New Customers": [
            "Stel geautomatiseerde e-mailreeksen in voor onboarding",
            "Bied een korting voor hun tweede aankoop",
            "Bied educatieve inhoud over uw producten of merk"
        ],
        "Promising": [
            "Stuur gepersonaliseerde aanbiedingen op basis van browsegedrag",
            "Stuur follow-up e-mails na aankoop",
            "Bied kortingen op verlaten winkelwagens"
        ],
        "Need Attention": [
            "Stuur reactivatie-e-mails met speciale kortingen",
            "Toon recensies, getuigenissen of sociaal bewijs",
            "Gebruik promoties met beperkte tijd om aankopen aan te moedigen"
        ],
        "About to Sleep": [
            "Stuur een 'We missen je' e-mail met een agressieve korting",
            "Bied gratis verzending als stimulans",
            "Stuur een reactivatie-enquête"
        ],
        "At Risk": [
            "Stuur gerichte e-mails die wijzigingen of nieuwe functies benadrukken",
            "Voer betrokkenheidsonderzoeken uit voor feedback",
            "Bied sterke prikkels zoals 30-50% korting op volgende aankoop"
        ],
        "Can't Lose Them": [
            "Bereik uit met dringende re-engagementcampagnes",
            "Bied exclusieve VIP-aanbiedingen",
            "Voer directe enquêtes of gesprekken om behoeften te begrijpen"
        ],
        "Hibernating": [
            "Stuur een laatste 'Kom terug' e-mail met een significante korting",
            "Overweeg om ze te verwijderen uit frequente marketing e-mails",
            "Voer tests uit met verschillende inhoudstypen voor re-engagement"
        ]
    }
    return recommendations.get(segment, ["Geen specifieke aanbevelingen beschikbaar."])

def generate_personalized_campaigns(segment):
    campaigns = {
        "Champions": ["VIP exclusieve productlancering", "Brand ambassador programma"],
        "Loyal Customers": ["Loyaliteitsprogramma tier upgrade", "Gepersonaliseerd productbundel"],
        "Potential Loyalists": ["Tijdelijke aanbieding op favoriete categorieën", "Vroege toegang tot uitverkoop"],
        "New Customers": ["Welkomstserie met producteducatie", "Eerste aankoop jubileumaanbieding"],
        "Promising": ["Categoriespecifieke kortingen", "Betrokkenheid-gedreven beloningen"],
        "Need Attention": ["Reactivatieserie met incrementele aanbiedingen", "Feedback-enquête met stimulans"],
        "About to Sleep": ["Laatste kans aanbieding", "Product aanbevelingsquiz"],
        "At Risk": ["Win-back campagne met hoge waarde stimulans", "Persoonlijk contact van klantenservice"],
        "Can't Lose Them": ["VIP conciërge service", "Exclusieve offline evenement uitnodiging"],
        "Hibernating": ["Herintroductie van nieuwe producten/functies", "Verras en verblij campagne"]
    }
    return campaigns.get(segment, ["Geen specifieke campagnes beschikbaar."])

def generate_dummy_data(n_customers=1000):
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=365*2)  # 2 years of data
    
    data = []
    for _ in range(n_customers):
        n_orders = np.random.randint(1, 50)
        customer_id = f"CUST_{np.random.randint(10000, 99999)}"
        for _ in range(n_orders):
            order_date = np.random.choice(dates)
            amount = np.random.uniform(10, 1000)
            data.append([customer_id, order_date, amount])
    
    return pd.DataFrame(data, columns=['customer_id', 'order_date', 'amount'])

def score_rfm_column(column, reverse=False):
    try:
        labels = range(5, 0, -1) if reverse else range(1, 6)
        return pd.qcut(column, q=5, labels=labels, duplicates='drop')
    except ValueError:
        return pd.cut(column, bins=5, labels=labels, duplicates='drop')

def map_columns(df, customer_id_col, order_date_col, amount_col):
    df = df.rename(columns={
        customer_id_col: 'customer_id',
        order_date_col: 'order_date',
        amount_col: 'amount'
    })
    return df

def validate_data(df, date_format):
    errors = []
    if df['customer_id'].isnull().any():
        errors.append("Er zijn ontbrekende waarden in de klant-ID kolom.")
    if df['order_date'].isnull().any():
        errors.append("Er zijn ontbrekende waarden in de orderdatum kolom.")
    if df['amount'].isnull().any():
        errors.append("Er zijn ontbrekende waarden in de bedrag kolom.")
    
    try:
        df['order_date'] = pd.to_datetime(df['order_date'], format=date_format)
    except:
        errors.append(f"Kon de orderdatum niet converteren naar een datum formaat met het opgegeven formaat: {date_format}")
    
    try:
        df['amount'] = pd.to_numeric(df['amount'])
    except:
        errors.append("Kon het bedrag niet converteren naar een numeriek formaat.")
    
    return errors

# Streamlit app
st.set_page_config(page_title="E-commerce RFM Analytics Suite", page_icon="📊", layout="wide")

st.title('🚀 E-commerce RFM Analytics Suite')
st.markdown("""
Deze krachtige analytics tool is ontworpen voor e-commerce bedrijven om hun klantgegevens te analyseren met RFM (Recency, Frequency, Monetary) analyse.
""")

# Data Source Selection
data_source = st.radio(
    "Kies uw gegevensbron:",
    ('Upload CSV', 'Gebruik Voorbeelddata')
)

df = None

if data_source == 'Upload CSV':
    uploaded_file = st.file_uploader("Upload uw klantdata CSV-bestand", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Voorbeeld van geüploade data:")
        st.write(df.head())
        
        # Kolomtoewijzing
        st.subheader("Kolomtoewijzing")
        customer_id_col = st.selectbox("Selecteer de kolom voor Klant ID:", df.columns)
        order_date_col = st.selectbox("Selecteer de kolom voor Orderdatum:", df.columns)
        amount_col = st.selectbox("Selecteer de kolom voor Orderbedrag:", df.columns)
        
        # Datumformaat specificatie
        date_format = st.text_input("Specificeer het datumformaat (bijv. %Y-%m-%d voor YYYY-MM-DD):", "%Y-%m-%d")
        st.info("Veel voorkomende datumformaten: %d-%m-%Y (31-12-2021), %m/%d/%Y (12/31/2021), %Y-%m-%d (2021-12-31)")
        
        if st.button("Bevestig kolomtoewijzing en datumformaat"):
            df = map_columns(df, customer_id_col, order_date_col, amount_col)
            st.success("Kolommen zijn toegewezen en datumformaat is ingesteld!")

elif data_source == 'Gebruik Voorbeelddata':
    st.info("Voorbeelddata wordt gebruikt voor demonstratiedoeleinden.")
    df = generate_dummy_data()
    date_format = "%Y-%m-%d"  # Standaard datumformaat voor voorbeelddata

if df is not None:
    # Data Validatie
    errors = validate_data(df, date_format)
    if errors:
        st.error("Er zijn problemen gevonden in de data:")
        for error in errors:
            st.write(f"- {error}")
    else:
        st.success("Data validatie succesvol!")

        # RFM Analyse
        current_date = datetime.now()
        rfm = df.groupby('customer_id').agg({
            'order_date': lambda x: (current_date - pd.to_datetime(x.max())).days,
            'customer_id': 'count',
            'amount': 'sum'
        })
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # RFM Scoring
        rfm['R_Score'] = score_rfm_column(rfm['Recency'], reverse=True)
        rfm['F_Score'] = score_rfm_column(rfm['Frequency'])
        rfm['M_Score'] = score_rfm_column(rfm['Monetary'])
        rfm['RFM_Score'] = rfm.apply(calculate_rfm_score, axis=1)
        rfm['Customer_Segment'] = rfm['RFM_Score'].apply(assign_customer_segment)

        # Dashboard
        st.header("📊 RFM Analyse Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RFM Score Verdeling")
            fig_rfm = px.histogram(rfm, x='RFM_Score', nbins=20, 
                                   labels={'RFM_Score': 'RFM Score', 'count': 'Aantal Klanten'},
                                   title='Verdeling van RFM Scores')
            st.plotly_chart(fig_rfm)
        
        with col2:
            st.subheader("Klantsegment Verdeling")
            segment_counts = rfm['Customer_Segment'].value_counts()
            fig_segments = px.pie(values=segment_counts.values, names=segment_counts.index, 
                                  title='Klantsegmenten')
            st.plotly_chart(fig_segments)

        st.subheader("RFM Segmentatie Bubble Chart")
        fig_bubble = px.scatter(rfm, x='Recency', y='Frequency', size='Monetary', color='Customer_Segment',
                                hover_name=rfm.index, size_max=60,
                                labels={'Recency': 'Recency (dagen)', 'Frequency': 'Frequency (aantal)', 'Monetary': 'Monetary (totale uitgaven)'},
                                title='RFM Segmentatie')
        st.plotly_chart(fig_bubble)

        # Actionable Insights
        st.header("🎯 Actie-inzichten")
        selected_segment = st.selectbox("Kies een klantsegment voor gepersonaliseerde strategieën:", 
                                        rfm['Customer_Segment'].unique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Aanbevolen Strategieën")
            strategies = get_segment_recommendations(selected_segment)
            for strategy in strategies:
                st.write(f"• {strategy}")
        
        with col2:
            st.subheader("Gepersonaliseerde Campagne-ideeën")
            campaigns = generate_personalized_campaigns(selected_segment)
            for campaign in campaigns:
                st.write(f"• {campaign}")

        # Next Steps
        st.header("💡 Volgende Stappen om Uw Omzet te Verhogen")
        st.markdown("""
        1. **Implementeer Gesegmenteerde Campagnes**: Gebruik de gepersonaliseerde campagne-ideeën om gerichte marketinginspanningen te creëren.
        2. **Focus op Klanten met Hoge CLV**: Identificeer en koester relaties met uw meest waardevolle klanten.
        3. **Verminder Churn**: Ontwikkel retentiestrategieën voor segmenten met een hoge kans op churn.
        4. **Optimaliseer de Klantreis**: Gebruik RFM-inzichten om uw algehele klantervaring te verbeteren.
        5. **Regelmatige Analyse**: Plan regelmatige RFM-analyses om de impact van uw strategieën in de loop van de tijd te volgen.
        """)

        # Download options
        st.header("📥 Download Uw Inzichten")
        
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