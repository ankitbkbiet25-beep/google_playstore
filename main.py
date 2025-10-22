import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from model_resources.BasicPreprocessing import basic_preprocess
import joblib

dataset=pd.read_csv("googleplaystore.csv")
df=basic_preprocess(dataset)
st.set_page_config(page_title="Google Playstore Dashboard", layout="centered")
st.title("Google Playstore Dashboard")
st.subheader("Apps insights and Rating Predictor")


insights,prediction=st.tabs(["Data Insights","Prediction"])
with insights:
    st.subheader("Data Insights")
    st.markdown("Key Metrics")
    
    def metric_card(label, value):
            st.markdown(f"""
            <div style="
                background: rgba(135, 206, 150, 0.15);
                border: 1px solid rgba(70, 130, 180, 0.3);
                border-radius: 18px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05);
                min-width: 200px;
                height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                margin: 10px 8px;
            ">
                <div style="font-size:14px; color:#4682B4; font-weight:500;">{label}</div>
                <div style="font-size:18px; color:#1E90FF; font-weight:600;">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    kpi1,kpi2,kpi3=st.columns(3)
    with kpi1:
        metric_card("Total Apps",len(df))
    with kpi2:
        avg_rating=df['Rating'].mean()
        metric_card("Average Rating",round(avg_rating,2))
    with kpi3:
        most_reviewed = df.loc[df["Reviews"].idxmax(),"App"]
        metric_card("Most Reviewed App",most_reviewed)
    
    kpi4,kpi5,kpi6=st.columns(3)
    with kpi4:
        most_install=df.loc[df["Installs"].idxmax(),"App"]
        metric_card("Most Installed App",most_install)
    with kpi5:
        common_category=df["Category"].mode()[0]
        metric_card("Most Common Category",common_category)
    with kpi6:
        free_apps_pct = (df['Type'] == 'Free').mean() * 100
        metric_card("Free Apps (%)", f"{free_apps_pct:.2f}%")
        
        
    distribution,relation=st.tabs(["Dataset Distribution","Features Relation"])
    with distribution:
        
        st.markdown(f"**◉ Dataset shape : `{df.shape[0]}` rows x `{df.shape[1]}` columns**")
        st.write("◉ **Column types:**")
        st.write(df.dtypes.value_counts())
        st.markdown(f"◉ Dataset Description ")
        st.table(df.describe().T)
        st.markdown("<h3 style='text-align: center;'> Distribution & Boxplots</h3>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left;'> ◉ Numerical Features Visulaization</h3>", unsafe_allow_html=True)
        
        numeric_cols = ['Rating', 'Reviews', 'Size', 'Installs', 'Price', 'Update_Year', 'update_month']
        for col in numeric_cols:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Distribution of {col}**")
                fig1, ax = plt.subplots(figsize=(5, 3.6))
                sns.histplot(df[col],kde=True,bins=30,color='skyblue',ax=ax)
                ax.axvline(df[col].mean(),color='red',linestyle='--',label='Mean')
                ax.axvline(df[col].median(),color='green',linestyle='-',label="Median")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig1)

            with col2:
                st.markdown(f"**Boxplot of {col}**")
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                sns.boxplot(x=df[col], ax=ax2, color='lightgreen')
                ax2.grid(True)
                st.pyplot(fig2)
                
        st.markdown("<h3 style='text-align: left;'> ◉ Categorical Features Visulaization</h3>", unsafe_allow_html=True)

        col1,col2,col3=st.columns(3)
        with col1:
            st.markdown("**Distribution of Payment Type**")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.countplot(x=df['Type'],order=df["Type"].value_counts().index,ax=ax,color='skyblue')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Distribution of Content Rating**")
            fig, ax = plt.subplots(figsize=(5, 3.23))
            sns.countplot(x=df['Content Rating'],order=df['Content Rating'].value_counts().index,ax=ax,color='skyblue')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        with col3:
            st.markdown("**Distribution of Secondary_genre**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(x=df['Secondary_genre'],order=df['Secondary_genre'].value_counts().index,ax=ax,color='skyblue')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
            
        top_categories = df['Category'].value_counts().nlargest(10).index
        df['Category_grouped'] = df['Category'].apply(lambda x: x if x in top_categories else 'Other')

        top_genres = df['Primary_genre'].value_counts().nlargest(10).index
        df['Primary_genre_grouped'] = df['Primary_genre'].apply(lambda x: x if x in top_genres else 'Other')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 10 Categories (Grouped)**")
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            sns.countplot(x=df['Category_grouped'], order=df['Category_grouped'].value_counts().index, ax=ax3, palette='Blues_d')
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
            st.pyplot(fig3)

        with col2:
            st.markdown("**Top 10 Primary Genres (Grouped)**")
            fig4, ax4 = plt.subplots(figsize=(5, 3))
            sns.countplot(x=df['Primary_genre_grouped'], order=df['Primary_genre_grouped'].value_counts().index, ax=ax4, palette='Blues_d')
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=90)
            st.pyplot(fig4)
    
    with relation:
        num_col=['Rating', 'Reviews', 'Size', 'Installs', 'Price', 'Update_Year', 'update_month']
        corr_matrix=df[num_col].corr(method='pearson')
        st.markdown("<h4 style='text-align: center;'>Numerical Feature Correlation Heatmap</h4>", unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,6))
        sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt='.2f',linewidth=0.5,ax=ax)
        st.pyplot(fig)
        
        top_categories = df['Category'].value_counts().nlargest(10).index
        top_genres = df['Primary_genre'].value_counts().nlargest(10).index

        category_grouped = df['Category'].apply(lambda x: x if x in top_categories else 'Other')
        genre_grouped = df['Primary_genre'].apply(lambda x: x if x in top_genres else 'Other')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h6 style='text-align: center;'>Average Rating by Genre and App Type</h6>", unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            sns.barplot(x=genre_grouped, y='Rating', hue='Type', data=df, estimator=np.mean, palette='Blues')
            ax1.set_xticklabels(ax3.get_xticklabels(), rotation=45)
            st.pyplot(fig1)

        with col2:
            st.markdown("<h6 style='text-align: center;'>Rating Distribution by Content Rating</h6>", unsafe_allow_html=True)
            fig2, ax4 = plt.subplots(figsize=(10, 5))
            sns.boxplot(x='Content Rating', y='Rating', data=df, palette='Blues')
            st.pyplot(fig2)

        def bucket_installs(val):
            if val <= 100:
                return "1–100"
            elif val <= 500:
                return "101–500"
            elif val <= 1000:
                return "501–1K"
            elif val <= 5000:
                return "1K–5K"
            elif val <= 10000:
                return "5K–10K"
            elif val <= 50000:
                return "10K–50K"
            elif val <= 100000:
                return "50K–100K"
            elif val <= 500000:
                return "100K–500K"
            elif val <= 1000000:
                return "500K–1M"
            elif val <= 5000000:
                return "1M–5M"
            else:
                return ">5M"

        installs_bucketed = df['Installs'].apply(bucket_installs)
        install_order = ["1–100", "101–500", "501–1K", "1K–5K", "5K–10K","10K–50K", "50K–100K", "100K–500K", "500K–1M","1M–5M", ">5M"]

        st.markdown("<h5 style='text-align: center;'>Rating Distribution by Install Buckets and App Type</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(x=installs_bucketed, y=df['Rating'], hue=df['Type'], order=install_order, palette='coolwarm', split=True, inner='quartile')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

        st.markdown("<h5 style='text-align: center;'>Update Year vs Rating (by Month)</h5>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(18, 8))
        sns.stripplot(x='Update_Year', y='Rating', hue='update_month', data=df, palette='coolwarm', alpha=0.6)
        st.pyplot(fig2)
        
with prediction:
    pipeline=joblib.load("model_resources/rating_prediction_pipeline.pkl")
    metrics_df=pd.read_csv("model_resources/model_metrics_table.csv")
    
    st.subheader("App Rating Class Prediction")
    with st.form("Fill the App data entries"):
        Reviews=st.number_input("Reviews",min_value=0,value=1000)
        Size=st.number_input("Size (MB)",min_value=0.0,value=20.0)
        Installs=st.number_input("Installs",min_value=0,value=100000)
        month=st.slider("Last Updated Month",min_value=1,max_value=12,value=6)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        Category_grouped=st.selectbox("Category",options=[ 'MEDICAL', 'PERSONALIZATION', 'GAME', 'LIFESTYLE','COMMUNICATION', 'FAMILY', 'PHOTOGRAPHY', 'TOOLS', 'PRODUCTIVITY','FINANCE', 'other'])
        primary_genre_grouped=st.selectbox("Primary Genre", options=['Personalization', 'Lifestyle', 'Communication','Photography', 'Tools', 'Education', 'Action', 'Entertainment','Productivity', 'Finance','other' ])
        Content_Rating_Grouped=st.selectbox('Content Rating',options=["Everyone","Upto 18","Above 18","Unrated"])
        is_paid_label = st.radio("App Type", options=["Free", "Paid"])
        is_paid = 1 if is_paid_label == "Paid" else 0
        reviews_density=Reviews/(Size + 1e-5)
        install_to_reviews=Installs/(Reviews + 1e-5)
        update_angle=np.arctan2(month_sin,month_cos)
        
        submitted=st.form_submit_button("Predict")
    
    if(submitted):
        input_data=pd.DataFrame([{
            "Reviews": Reviews,
            "Size": Size,
            "Installs": Installs,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "Category_grouped": Category_grouped,
            "primary_genre_grouped": primary_genre_grouped,
            "Content_Rating_Grouped": Content_Rating_Grouped,
            "review_density": reviews_density,
            "install_to_review": install_to_reviews,
            "update_angle": update_angle,
            "is_paid": is_paid
        }])
        prediction=pipeline.predict(input_data)[0]
        prob=pipeline.predict_proba(input_data)[0]
        st.success(f"Predicted Rating Class: {'High' if prediction == 1 else 'Low'}")
        st.write(f"Confidence: {max(prob):.2%}")
    metrics,path=st.tabs(["Metrics","Modeling Timeline"])
    with metrics:
        st.header("Final Model Performance")
        st.dataframe(metrics_df.style.format({"Value": "{:.2%}"}))
        
    with path:
        st.subheader("How We Got Here")
        st.markdown("""
        ### Phase 1: Regression Attempt
        - Started with predicting app ratings as a continuous variable.
        - Used models like Linear Regression, Random Forest Regressor.
        - Achieved very low R² score (~0.08), indicating poor fit.
        - Tried various models as svr, GradientBoostingRegressor,etc
        - With Xgboost achieved highest R² as ~0.11
        - Explored feature engineering and transformations.
        - Reference: `googleplaystoreEDA.ipynb`

        ### Phase 2: Rethinking the Problem
        - Decided to reframe the task as **classification**: High vs Low rating.
        - Created binary target variable based on rating threshold.
        - Applied resampling techniques to handle class imbalance.

        ### Phase 3: Classification Modeling
        - Tried models like Logistic Regression, XGBoost, CatBoost, Random Forest.
        - Evaluated using accuracy, precision, recall, F1-score.
        - Reference: `alliin.ipynb`

        ### Final Phase: Ensemble Voting Classifier
        - Combined XGBoost, CatBoost, and Random Forest using soft voting.
        - Achieved strong performance and interpretability.
        - Saved pipeline and metrics for deployment.
        """)

        st.caption("This journey reflects iterative experimentation, learning from failures, and refining the approach to reach a robust solution.")