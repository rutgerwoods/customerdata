import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import io
from datetime import datetime, timedelta
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

# Utility functions
def hash_customer_id(customer_id):
    return hashlib.sha256(str(customer_id).encode()).hexdigest()

# Wijzig deze functie
def calculate_rfm_score(row):
    try:
        return int(f"{int(row['R_Score'])}{int(row['F_Score'])}{int(row['M_Score'])}")
    except ValueError:
        st.error(f"Error calculating RFM score for row: {row}")
        return 111  # Default score

def assign_customer_segment(score):
    if score >= 555:
        return "Topklanten"
    elif 445 <= score <= 554:
        return "Trouwe Llanten"
    elif 334 <= score <= 444:
        return "Potentiële Loyalisten"
    elif 511 <= score <= 555:
        return "Nieuwe Klanten"
    elif 412 <= score <= 444:
        return "Veelbeloend"
    elif 322 <= score <= 411:
        return "Behoeft Aandacht"
    elif 311 <= score <= 322:
        return "Bijna Inactief"
    elif 222 <= score <= 310:
        return "Risico"
    elif 145 <= score <= 311:
        return "Kunnen We Niet Verliezen"
    else:
        return "Slapend"

def get_segment_recommendations(segment):
    recommendations = {
        "Topklanten": [
            "Bied vroege toegang tot nieuwe producten of exclusieve aanbiedingen",
            "Geef extra voordelen in uw loyaliteitsprogramma",
            "Moedig doorverwijzingen aan met beloningen"
        ],
        "Trouwe Klanten": [
            "Stuur gepersonaliseerde productaanbevelingen",
            "Gebruik gerichte e-mailcampagnes voor upselling en cross-selling",
            "Bied beloningen voor recensies of engagement op sociale media"
        ],
        "Potentiële Loyalisten": [
            "Bied tijdgebonden kortingen of gratis verzending bij de volgende aankoop",
            "Stuur gratis monsters van nieuwe of premium producten",
            "Vraag om feedback of recensies over recente aankopen"
        ],
        "Nieuwe Klanten": [
            "Zet geautomatiseerde e-mailreeksen op voor onboarding",
            "Bied korting aan voor hun tweede aankoop",
            "Verstrek educatieve inhoud over uw producten of merk"
        ],
        "Veelbelovend": [
            "Stuur gepersonaliseerde aanbiedingen op basis van surfgedrag",
            "Stuur follow-up e-mails na aankoop",
            "Bied kortingen aan op verlaten winkelwagens"
        ],
        "Behoeft Aandacht": [
            "Stuur reactivatie-e-mails met speciale kortingen",
            "Toon recensies, getuigenissen of sociaal bewijs",
            "Gebruik promoties met beperkte tijd om aankopen te stimuleren"
        ],
        "Bijna Inactief": [
            "Stuur een 'We missen u' e-mail met een agressieve korting",
            "Bied gratis verzending aan als stimulans",
            "Stuur een reactivatie-enquête"
        ],
        "Risico": [
            "Stuur gerichte e-mails die wijzigingen of nieuwe functies benadrukken",
            "Voer betrokkenheidsenquêtes uit voor feedback",
            "Bied sterke prikkels zoals 30-50% korting op de volgende aankoop"
        ],
        "Kunnen We Niet Verliezen": [
            "Benader met urgente heractiveringscampagnes",
            "Bied exclusieve VIP-aanbiedingen",
            "Voer directe enquêtes of gesprekken om behoeften te begrijpen"
        ],
        "Slapend": [
            "Stuur een laatste 'Kom terug' e-mail met een aanzienlijke korting",
            "Overweeg om ze te verwijderen uit frequente marketing e-mails",
            "Voer tests uit met verschillende inhoudstypen voor heractivering"
        ]
    }
    return recommendations.get(segment, ["Geen specifieke aanbevelingen beschikbaar."])

def fetch_shopify_data(api_key, shop_url):
    # Placeholder function - replace with actual Shopify API integration
    st.warning("Dit is een placeholder voor de Shopify integratie. Bij een echte implementatie zou dit data ophalen uit je Shopify account.")
    return pd.DataFrame()  # Return empty DataFrame for now

def fetch_mailchimp_data(api_key, list_id):
    # Placeholder function - replace with actual Mailchimp API integration
    st.warning("Dit is een placeholder voor de Mailchimp integratie. Bij een echte implementatie zou dit data ophalen uit je Mailchimp account.")
    return pd.DataFrame()  # Return empty DataFrame for now

def calculate_advanced_clv(df, time_period=24):
    # Prepare the data
    df['order_date'] = pd.to_datetime(df['order_date'])
   
    # Filter out rows with non-positive monetary values
    df_positive = df[df['amount'] > 0]
   
    if len(df_positive) < len(df):
        st.warning(f"Removed {len(df) - len(df_positive)} transactions with non-positive monetary values.")
   
    summary = summary_data_from_transaction_data(df_positive, 'customer_id', 'order_date', 'amount')
   
    # Data checks
    if summary['frequency'].max() == 0:
        st.error("Data issue: All customers have only one purchase. CLV calculation is not possible.")
        return None, None, None

    # Remove customers with extreme values
    for col in ['frequency', 'recency', 'T', 'monetary_value']:
        low = summary[col].quantile(0.01)
        high = summary[col].quantile(0.99)
        summary = summary[(summary[col] >= low) & (summary[col] <= high)]

    # Ensure all monetary values are positive
    summary['monetary_value'] = summary['monetary_value'].clip(lower=0.01)

    # Log summary statistics
    st.info(f"Analyzing {len(summary)} customers after data preparation.")
    st.info(f"Monetary value range: {summary['monetary_value'].min():.2f} to {summary['monetary_value'].max():.2f}")

    # Fit the BG/NBD model with higher penalizer
    bgf = BetaGeoFitter(penalizer_coef=0.1)
    try:
        bgf.fit(summary['frequency'], summary['recency'], summary['T'])
    except Exception as e:
        st.error(f"Error fitting BG/NBD model: {str(e)}")
        return None, None, None

    # Fit the Gamma-Gamma model with higher penalizer
    ggf = GammaGammaFitter(penalizer_coef=0.1)
    try:
        # Only use customers with at least one repeat purchase
        repeat_customers = summary[summary['frequency'] > 0]
        ggf.fit(repeat_customers['frequency'], repeat_customers['monetary_value'])
    except Exception as e:
        st.error(f"Error fitting Gamma-Gamma model: {str(e)}")
        return None, None, None

    # Predict future transactions
    try:
        predicted_purchases = bgf.predict(time_period, summary['frequency'], summary['recency'], summary['T'])
    except Exception as e:
        st.error(f"Error predicting future purchases: {str(e)}")
        return None, None, None

    # Calculate CLV
    try:
        clv = ggf.customer_lifetime_value(
            bgf,
            summary['frequency'],
            summary['recency'],
            summary['T'],
            summary['monetary_value'],
            time=time_period,
            discount_rate=0.01
        )
    except Exception as e:
        st.error(f"Error calculating CLV: {str(e)}")
        return None, None, None

    # Combine results
    results = pd.concat([summary, clv.rename('CLV')], axis=1)
    results['predicted_purchases'] = predicted_purchases

    # Additional calculations for comparison
    average_clv = clv.mean()
    total_revenue = df['amount'].sum()
    total_customers = df['customer_id'].nunique()
    average_revenue_per_customer = total_revenue / total_customers

    st.info(f"Gemiddelde CLV: €{average_clv:.2f}")
    st.info(f"Gemiddelde omzet per klant: €{average_revenue_per_customer:.2f}")
    st.info(f"Totale omzet: €{total_revenue:.2f}")
    st.info(f"Totaal aantal unieke klanten: {total_customers}")
    st.info(f"Gemiddeld aantal aankopen per klant: {df.groupby('customer_id').size().mean():.2f}")
    st.info(f"Gemiddelde tijd sinds eerste aankoop (in dagen): {summary['T'].mean():.2f}")

    # Add comparison information to results
    results['Average_Revenue'] = average_revenue_per_customer
    results['CLV_to_Average_Revenue_Ratio'] = results['CLV'] / average_revenue_per_customer
   
    return results, bgf, ggf

def visualize_clv(results, bgf, ggf):
    if results is None or bgf is None or ggf is None:
        st.error("Unable to visualize CLV due to calculation errors.")
        return

    st.subheader("Customer Lifetime Value Analyse")
   
    # CLV Distribution
    fig_clv = px.histogram(results, x='CLV', nbins=50,
                           title='Verdeling van Customer Lifetime Value',
                           labels={'CLV': 'Customer Lifetime Value'})
    st.plotly_chart(fig_clv)
   
    # Top 10 Customers by CLV
    top_customers = results.sort_values('CLV', ascending=False).head(10)
    st.subheader("Top 10 Klanten bij Lifetime Value")
    st.write(top_customers[['frequency', 'recency', 'T', 'monetary_value', 'CLV']])

    # CLV vs Average Revenue Comparison
    fig_comparison = px.scatter(results, x='Average_Revenue', y='CLV',
                                title='CLV vs. Gemiddelde Omzet per Klant',
                                labels={'Average_Revenue': 'Gemiddelde Omzet', 'CLV': 'Customer Lifetime Value'},
                                hover_data=['customer_id', 'frequency', 'recency'])
    st.plotly_chart(fig_comparison)

    # CLV to Average Revenue Ratio Distribution
    fig_ratio = px.histogram(results, x='CLV_to_Average_Revenue_Ratio',
                             title='Verdeling van CLV / Gemiddelde Omzet Ratio',
                             labels={'CLV_to_Average_Revenue_Ratio': 'CLV / Gemiddelde Omzet Ratio'})
    st.plotly_chart(fig_ratio)

def calculate_customer_lifetime_value(df):
    return df.groupby('customer_id')['amount'].sum()

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
    return campaigns.get(segment, ["Geen specifieke campagne beschikbaar."])

def generate_dummy_data(n_customers=1000, start_date=None, end_date=None):
    if start_date is None:
        start_date = datetime.now() - timedelta(days=730)  # 2 years ago
    if end_date is None:
        end_date = datetime.now()

    np.random.seed(42)
    date_range = pd.date_range(start=start_date, end=end_date)
    
    data = []
    for _ in range(n_customers):
        n_orders = np.random.randint(1, 20)  # Reduced max orders for more variability
        customer_id = f"CUST_{np.random.randint(10000, 99999)}"
        for _ in range(n_orders):
            order_date = np.random.choice(date_range)
            # Ensure strictly positive amounts with some variability
            amount = max(0.01, np.random.lognormal(mean=3, sigma=1))  # Minimum amount of 0.01
            data.append([customer_id, order_date, amount])
    
    df = pd.DataFrame(data, columns=['customer_id', 'order_date', 'amount'])
    
    # Sort by customer_id and order_date
    df = df.sort_values(['customer_id', 'order_date'])
    
    # Ensure no duplicate (customer_id, order_date) combinations
    df = df.groupby(['customer_id', 'order_date']).agg({'amount': 'sum'}).reset_index()
    
    # Extra check to ensure all amounts are positive
    assert (df['amount'] > 0).all(), "Error: Non-positive amounts found in generated data"
    
    return df

# Optioneel: Voeg deze functie toe om de gegenereerde data te controleren
def check_generated_data(df):
    print(f"Total number of records: {len(df)}")
    print(f"Number of unique customers: {df['customer_id'].nunique()}")
    print(f"Date range: {df['order_date'].min()} to {df['order_date'].max()}")
    print(f"Amount range: {df['amount'].min():.2f} to {df['amount'].max():.2f}")
    print(f"Any non-positive amounts: {(df['amount'] <= 0).any()}")

# Gebruik:
# df = generate_dummy_data()
# check_generated_data(df)

# Wijzig deze functie
def score_rfm_column(column, reverse=False):
    try:
        labels = range(5, 0, -1) if reverse else range(1, 6)
        return pd.qcut(column, q=5, labels=labels, duplicates='drop')
    except ValueError:
        # Als qcut faalt vanwege te veel dubbele waarden, gebruik dan een eenvoudigere methode
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


# Aangepaste exportfunctie
def export_segment_data(df, rfm, segment=None):
    export_df = df.merge(rfm[['Customer_Segment']], left_on='customer_id', right_index=True)
    
    if segment and segment != "Alle segmenten":
        export_df = export_df[export_df['Customer_Segment'] == segment]

    export_columns = ['customer_id', 'order_date', 'amount', 'Customer_Segment']
    export_df = export_df[export_columns].drop_duplicates()
    
    csv = export_df.to_csv(index=False)
    return csv


# Nieuwe functie voor het berekenen van metrics
def calculate_metrics(df):
    total_customers = df['customer_id'].nunique()
    total_revenue = df['amount'].sum()
    average_revenue = total_revenue / total_customers
    aov = df['amount'].mean()
    avg_purchases = df.groupby('customer_id').size().mean()
    clv = total_revenue / total_customers  # Simple CLV calculation

    return total_customers, total_revenue, average_revenue, aov, avg_purchases, clv

def create_home_dashboard(df, rfm):
    st.header("🏠 Dashboard")

    # Calculate basic metrics
    total_customers, total_revenue, average_revenue, aov, avg_purchases, simple_clv = calculate_metrics(df)

    # Calculate advanced CLV
    results, _, _ = calculate_advanced_clv(df)
    
    if results is not None and 'CLV' in results.columns:
        advanced_clv = results['CLV'].mean()
    else:
        advanced_clv = simple_clv
        st.warning("Geavanceerde CLV calculatie gestopt. Gebruik de simpele CLV.")

    # KPI overzicht
    col1, col2, col3 = st.columns(3)
    col1.metric("Totaal Aantal Klanten", f"{total_customers:,}")
    col2.metric("Totale Omzet", f"€{total_revenue:,.2f}")
    col3.metric("Gemiddelde Omzet per Klant", f"€{average_revenue:,.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Gemiddelde Orderwaarde (AOV)", f"€{aov:,.2f}")
    col5.metric("Gemiddeld Aantal Aankopen", f"{avg_purchases:.2f}")
    col6.metric("Gemiddelde Customer Lifetime Value (CLV)", f"€{advanced_clv:.2f}")


    # Totale uitgaven per segment
    st.subheader("Totale Uitgaven per Segment")
    segment_spend = rfm.groupby('Customer_Segment')['Monetary'].sum().reset_index()
    segment_spend = segment_spend.sort_values('Monetary', ascending=False)
    fig_bar = px.bar(segment_spend, x='Customer_Segment', y='Monetary', 
                     title='Totale Uitgaven per Segment',
                     labels={'Monetary': 'Totale Uitgaven', 'Customer_Segment': 'Segment'})
    st.plotly_chart(fig_bar)

# Streamlit app
st.set_page_config(page_title="E-commerce RFM Analytics Suite", page_icon="📊", layout="wide")

st.title('🚀 E-commerce RFM Analytics Suite')
st.markdown("""
Deze krachtige analyse-tool is speciaal ontwikkeld voor e-commerce bedrijven. Benut de mogelijkheden van RFM-analyse (Recency, Frequency, Monetary) om uw marketingstrategie te versterken en uw omzet te vergroten.
""")

# Data Source Selection
data_source = st.radio(
    "Kies je databron:",
    ('Upload CSV', 'Connect Shopify', 'Connect Mailchimp', 'Gebruik Dummy Data')
)

df = None
date_format = "%Y-%m-%d"  # Standaard datumformaat

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

elif data_source == 'Connect Shopify':
    st.info("Shopify integratie komt eraan! Gebruik voor nu de CSV upload optie alstublieft.")
    api_key = st.text_input("Enter your Shopify API key")
    shop_url = st.text_input("Enter your Shopify store URL")
    if st.button("Fetch Shopify Data"):
        df = fetch_shopify_data(api_key, shop_url)
elif data_source == 'Connect Mailchimp':
    st.info("Mailchimp integratie komt eraan! Gebruik voor nu de CSV upload optie alstublieft.")
    api_key = st.text_input("Vul je Mailchimp API key in")
    list_id = st.text_input("Vul je Mailchimp list ID in")
    if st.button("Fetch Mailchimp Data"):
        df = fetch_mailchimp_data(api_key, list_id)
elif data_source == 'Gebruik Dummy Data':
    st.info("Gebruik Dummy Data voor demonstratie.")
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

        create_home_dashboard(df, rfm)

        # Dashboard
        st.header("📊 RFM Analyse Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RFM Score Verdeling")
            fig_rfm = px.histogram(rfm, x='RFM_Score', nbins=20, 
                                   labels={'RFM_Score': 'RFM Score', 'count': 'Number of Customers'},
                                   title='Distribution of RFM Scores')
            st.plotly_chart(fig_rfm)
        
        with col2:
            st.subheader("Customer Segment Verdeling")
            segment_counts = rfm['Customer_Segment'].value_counts()
            fig_segments = px.pie(values=segment_counts.values, names=segment_counts.index, 
                                  title='Customer Segments')
            st.plotly_chart(fig_segments)
        
        st.subheader("RFM Segmentatie Grafiek")
        fig_bubble = px.scatter(rfm, x='Recency', y='Frequency', size='Monetary', color='Customer_Segment',
                                hover_name=rfm.index, size_max=60,
                                labels={'Recency': 'Recency (days)', 'Frequency': 'Frequency (count)', 'Monetary': 'Monetary (total spend)'},
                                title='RFM Segmentation')
        st.plotly_chart(fig_bubble)

        # Advanced Analytics
        st.header("🧠 Geavanceerde Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Lifetime Value")
            clv = calculate_customer_lifetime_value(df)
            fig_clv = px.histogram(clv, nbins=20, labels={'value': 'Customer Lifetime Value'},
                                   title='Distribution of Customer Lifetime Value')
            st.plotly_chart(fig_clv)


        # In your main Streamlit app
        st.header("🔮 Geavanceerde Customer Lifetime Value Analyse")
        results, bgf, ggf = calculate_advanced_clv(df)
        if results is not None:
            visualize_clv(results, bgf, ggf)

            # Additional Insights
            st.subheader("CLV Inzichten")
            total_clv = results['CLV'].sum()
            avg_clv = results['CLV'].mean()
            st.write(f"Total Customer Lifetime Value: €{total_clv:,.2f}")
            st.write(f"Average Customer Lifetime Value: €{avg_clv:,.2f}")

            # Segment CLV Analysis
            if 'Customer_Segment' in results.columns:
                segment_clv = results.groupby('Customer_Segment')['CLV'].agg(['mean', 'sum']).sort_values('sum', ascending=False)
                st.write("CLV by Customer Segment:")
                st.write(segment_clv)
            else:
                st.warning("Customer Segment information is not available for CLV analysis.")

            
        else:
            st.warning("CLV calculation failed. Please check your data and try again.")

                # Actionable Insights
        st.header("🎯 Actionable Insights")
        selected_segment = st.selectbox("Kies een klantsegment voor gepersonaliseerde strategieën:", 
                                    rfm['Customer_Segment'].unique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Aanbevolen Strategieën")
            strategies = get_segment_recommendations(selected_segment)
            for strategy in strategies:
                st.write(f"• {strategy}")
        
        with col2:
            st.subheader("Gepersonaliseerde Campagne Ideeën")
            campaigns = generate_personalized_campaigns(selected_segment)
            for campaign in campaigns:
                st.write(f"• {campaign}")

        # Exporteer Segmentdata
        st.header("📤 Exporteer Segmentdata")
        st.write("Selecteer een segment om de klantgegevens te exporteren voor gebruik in e-mailmarketingtools:")

        # Voeg 'Alle segmenten' toe aan de lijst met segmenten
        segment_options = ['Alle segmenten'] + list(rfm['Customer_Segment'].unique())
        selected_segment = st.selectbox("Kies een segment om te exporteren:", segment_options)

        if st.button("Genereer Download"):
            csv = export_segment_data(df, rfm, selected_segment)
            
            if selected_segment == "Alle segmenten":
                filename = "alle_klanten_segmenten.csv"
            else:
                filename = f"{selected_segment.lower().replace(' ', '_')}_klanten.csv"
            
            st.download_button(
                label=f"Download {selected_segment} Data",
                data=csv,
                file_name=filename,
                mime="text/csv",
            )

        # Voeg wat uitleg toe over het gebruik van de geëxporteerde data
        st.info("""
        ### Hoe te importeren in e-mailmarketingtools:
        1. **Mailchimp**: Ga naar Audience > Audience fields > Import contacts > CSV file
        2. **Klaviyo**: Ga naar Lists & Segments > Create list/segment > Upload CSV
        3. **HubSpot**: Ga naar Contacts > Import > Start an import > File from computer
        """)


        # Call-to-Action
        st.markdown("""
        ---
        ## 🚀 Klaar om uw e-commerce marketing naar een hoger niveau te tillen?
        Deze gratis tool is nog maar het begin. Stel u voor wat we kunnen bereiken met een volledige integratie van uw gegevensbronnen en op maat gemaakte strategieën.
       
        ### Boek een gratis consult om te ontdekken hoe we:
        - Volledig kunnen integreren met uw Shopify-winkel en Mailchimp-campagnes
        - Real-time RFM-analyse en geautomatiseerde segmentatie kunnen bieden
        - AI-gestuurde voorspellende modellen voor klantgedrag kunnen ontwikkelen
        - Een dashboard op maat kunnen creëren, afgestemd op uw specifieke KPI's
       
        Mail naar rutger.bos@dataoaklabs.com om een afspraak te maken!
        """)
else:
    st.info("Upload uw gegevens of maak verbinding met een gegevensbron om te beginnen!")

# Footer
st.markdown("""
---
Made with ❤️ by Oak Data Labs
    """)