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

def calculate_advanced_clv(df, time_period=12):
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

    st.info(f"Analyzing {len(summary)} customers after data preparation.")

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
        ggf.fit(summary['frequency'], summary['monetary_value'])
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
    
    return results, bgf, ggf

def visualize_clv(results, bgf, ggf):
    if results is None or bgf is None or ggf is None:
        st.error("Unable to visualize CLV due to calculation errors.")
        return

    st.subheader("Customer Lifetime Value Analysis")
    
    # CLV Distribution
    fig_clv = px.histogram(results, x='CLV', nbins=50,
                           title='Distribution of Customer Lifetime Value',
                           labels={'CLV': 'Customer Lifetime Value'})
    st.plotly_chart(fig_clv)
    
    # CLV vs Frequency
    fig_freq = px.scatter(results, x='frequency', y='CLV', color='monetary_value',
                          title='CLV vs Purchase Frequency',
                          labels={'frequency': 'Purchase Frequency', 'CLV': 'Customer Lifetime Value'})
    st.plotly_chart(fig_freq)
    
    # Top 10 Customers by CLV
    top_customers = results.sort_values('CLV', ascending=False).head(10)
    st.subheader("Top 10 Customers by Lifetime Value")
    st.write(top_customers[['frequency', 'recency', 'T', 'monetary_value', 'CLV']])
    

# Update deze functie
def calculate_customer_lifetime_value(df):
    return df.groupby('customer_id')['amount'].sum()

def predict_churn(rfm, df, churn_threshold=90):
    # Define churn (e.g., customers who haven't purchased in X days)
    last_purchase_date = df.groupby('customer_id')['order_date'].max()
    last_purchase_date = pd.to_datetime(last_purchase_date)
    current_date = pd.to_datetime('today')
    rfm['Churn'] = ((current_date - last_purchase_date).dt.days > churn_threshold).astype(int)

    # Check if we have both churned and non-churned customers
    if rfm['Churn'].nunique() == 1:
        st.warning(f"All customers are classified as {'churned' if rfm['Churn'].iloc[0] == 1 else 'non-churned'}. "
                   f"Consider adjusting the churn threshold (currently {churn_threshold} days).")
        return rfm, None, None, None, None

    # Prepare features and target
    X = rfm[['Recency', 'Frequency', 'Monetary']]
    y = rfm['Churn']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    # Check if we have both classes in the predictions
    if len(np.unique(y_pred)) == 1:
        st.warning("The model predicted only one class. ROC AUC score cannot be calculated.")
        roc_auc = None
    else:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

    report = classification_report(y_test, y_pred)

    # Predict churn probability for all customers
    rfm['Churn_Probability'] = rf_model.predict_proba(X)[:, 1]

    return rfm, rf_model, accuracy, roc_auc, report


def visualize_churn(rfm, accuracy, roc_auc):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Probability Distribution")
        fig_churn = px.histogram(rfm, x='Churn_Probability', nbins=20,
                                 labels={'Churn_Probability': 'Churn Probability'},
                                 title='Distribution of Churn Probability')
        st.plotly_chart(fig_churn)

    with col2:
        st.subheader("Churn Risk by Segment")
        segment_churn = rfm.groupby('Customer_Segment')['Churn_Probability'].mean().sort_values(ascending=False)
        fig_segment_churn = px.bar(segment_churn, x=segment_churn.index, y='Churn_Probability',
                                   labels={'Churn_Probability': 'Average Churn Probability'},
                                   title='Average Churn Probability by Segment')
        st.plotly_chart(fig_segment_churn)

    st.subheader("Model Performance")
    st.write(f"Model Accuracy: {accuracy:.2f}")
    st.write(f"ROC AUC Score: {roc_auc:.2f}")

    # Feature Importance
    st.subheader("Feature Importance for Churn Prediction")
    feature_importance = pd.DataFrame({
        'feature': ['Recency', 'Frequency', 'Monetary'],
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    fig_importance = px.bar(feature_importance, x='feature', y='importance',
                            labels={'importance': 'Importance'},
                            title='Feature Importance in Churn Prediction')
    st.plotly_chart(fig_importance)

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
            # Ensure positive amounts with some variability
            amount = np.random.lognormal(mean=3, sigma=1)  # This generates positive values
            data.append([customer_id, order_date, amount])
    
    df = pd.DataFrame(data, columns=['customer_id', 'order_date', 'amount'])
    
    # Sort by customer_id and order_date
    df = df.sort_values(['customer_id', 'order_date'])
    
    # Ensure no duplicate (customer_id, order_date) combinations
    df = df.groupby(['customer_id', 'order_date']).agg({'amount': 'sum'}).reset_index()
    
    return df

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
    st.header("🏠 Home Dashboard")

    # Calculate basic metrics
    total_customers, total_revenue, average_revenue, aov, avg_purchases, simple_clv = calculate_metrics(df)

    # Calculate advanced CLV
    results, _, _ = calculate_advanced_clv(df)
    
    if results is not None and 'CLV' in results.columns:
        advanced_clv = results['CLV'].mean()
    else:
        advanced_clv = simple_clv
        st.warning("Advanced CLV calculation failed. Using simple CLV instead.")

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
This powerful analytics tool is designed exclusively for e-commerce businesses with 11-50 employees. 
Harness the power of RFM (Recency, Frequency, Monetary) analysis to supercharge your marketing strategy and skyrocket your revenue!
""")

# Data Source Selection
data_source = st.radio(
    "Choose your data source:",
    ('Upload CSV', 'Connect Shopify', 'Connect Mailchimp', 'Use Dummy Data')
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
elif data_source == 'Use Dummy Data':
    st.info("Using dummy data for demonstration purposes.")
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

    # In your main Streamlit app
    st.header("🔮 Churn Prediction")
    churn_threshold = st.slider("Select churn threshold (days)", min_value=30, max_value=365, value=90, step=30)
    rfm, rf_model, accuracy, roc_auc, report = predict_churn(rfm, df, churn_threshold)

    if rf_model is not None:
        visualize_churn(rfm, accuracy, roc_auc)

        # Display additional insights
        st.subheader("Top 10 Customers at Risk of Churn")
        high_risk_customers = rfm.sort_values('Churn_Probability', ascending=False).head(10)
        st.write(high_risk_customers[['Recency', 'Frequency', 'Monetary', 'Customer_Segment', 'Churn_Probability']])

        st.subheader("Churn Prediction Report")
        st.text(report)
    else:
        st.warning("Churn prediction could not be performed due to data limitations.")
        
        # In your main Streamlit app
        st.header("🔮 Advanced Customer Lifetime Value Analysis")
        results, bgf, ggf = calculate_advanced_clv(df)
        if results is not None:
            visualize_clv(results, bgf, ggf)

            # Additional Insights
            st.subheader("CLV Insights")
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

            # CLV Improvement Recommendations
            st.subheader("CLV Improvement Recommendations")
            st.write("""
            1. Focus on increasing purchase frequency for high-value customers
            2. Develop targeted retention strategies for segments with high CLV
            3. Implement personalized marketing campaigns based on predicted future value
            4. Optimize customer acquisition costs based on expected CLV
            5. Regularly update the CLV model to reflect changing customer behaviors
            """)
        else:
            st.warning("CLV calculation failed. Please check your data and try again.")
                    
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