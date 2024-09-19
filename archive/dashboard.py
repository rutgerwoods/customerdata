import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import io

def hash_customer_id(customer_id):
    return hashlib.sha256(str(customer_id).encode()).hexdigest()

# Titel van de app
st.title('RFM Analysis Dashboard')

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
            rfm = df.groupby('hashed_customer_id').agg({
                'order_date': lambda x: (pd.Timestamp.now() - x.max()).days,
                'hashed_customer_id': 'count',
                'amount': 'sum'
            })
            
            rfm.columns = ['Recency', 'Frequency', 'Monetary']
            
            # Bereken RFM-scores
            rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop')
            rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method="first"), 4, labels=[1, 2, 3, 4])
            rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4], duplicates='drop')
            
            rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
            
            # Toon RFM-analyse resultaten
            st.write("RFM Analyse Resultaten:")
            st.dataframe(rfm)
            
            # RFM Score Verdeling
            st.write("RFM Score Verdeling:")
            fig, ax = plt.subplots()
            rfm['RFM_Score'].value_counts().sort_index().plot(kind='bar', ax=ax)
            ax.set_xlabel('RFM Score')
            ax.set_ylabel('Aantal Klanten')
            ax.set_title('Aantal Klanten per RFM Score')
            st.pyplot(fig)
            
            # Heatmap van RFM-factoren
            st.write("Heatmap van RFM-factoren:")
            corr_matrix = rfm[['Recency', 'Frequency', 'Monetary']].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            
            # Download RFM Analyse Rapport
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                rfm.to_excel(writer, sheet_name='RFM Analysis', index=False)
            buf.seek(0)
            st.download_button(label="Download RFM Analyse Rapport", data=buf, file_name="rfm_analysis.xlsx")

    except Exception as e:
        st.error(f"Er is een fout opgetreden: {e}")
