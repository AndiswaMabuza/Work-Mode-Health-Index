import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(layout="wide")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("post_pandemic_remote_work_health_impact_2025.csv", parse_dates=["Survey_Date"])
    df["Num_Physical_Issues"] = df["Physical_Health_Issues"].fillna("None").apply(lambda x: 0 if x == "None" else len(x.split(";")))
    salary_map = {
        "$40K-60K": 50000, "$60K-80K": 70000, "$80K-100K": 90000,
        "$100K-120K": 110000, "$120K+": 130000
    }
    burnout_map = {"Low": 0, "Medium": 1, "High": 2}
    df["Salary_Estimate"] = df["Salary_Range"].map(salary_map)
    df["Burnout_Level_Encoded"] = df["Burnout_Level"].map(burnout_map)
    df["Age_Group"] = pd.cut(df["Age"], bins=[18, 30, 40, 50, 65], labels=["18-30", "31-40", "41-50", "51-65"])
    return df.dropna(subset=["Work_Arrangement", "Burnout_Level"])

df = load_data()
features = ["Work_Arrangement", "Job_Role", "Region", "Hours_Per_Week", "Social_Isolation_Score", "Num_Physical_Issues", "Salary_Estimate"]

# ---------- SIDEBAR ----------
st.sidebar.title("📌 Scenario Simulator")
selected_job = st.sidebar.selectbox("Select Job Role", df["Job_Role"].unique())
selected_region = st.sidebar.selectbox("Select Region", df["Region"].unique())
selected_work_modes = ["Onsite", "Remote", "Hybrid"]

# ---------- TABS ----------
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🧠 Diagnostics", "🤖 Predictive Model", "🔮 Scenario Simulation"])

# ---------- TAB 1: EDA ----------
with tab1:
    st.header("📊 Exploratory Data Analysis")
    fig = px.histogram(df, x="Region", color="Work_Arrangement", barmode="group", title="Work Arrangement by Region")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x="Age_Group", color="Work_Arrangement", barmode="group", title="Work Arrangement by Age Group")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")
    eda_cols = ["Burnout_Level_Encoded", "Hours_Per_Week", "Work_Life_Balance_Score", "Num_Physical_Issues", "Social_Isolation_Score", "Salary_Estimate"]
    corr = df[eda_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------- TAB 2: Diagnostics ----------
with tab2:
    st.header("🧠 ANOVA and Tukey HSD")
    model = ols("Burnout_Level_Encoded ~ C(Work_Arrangement)", data=df).fit()
    anova_result = sm.stats.anova_lm(model, typ=2)
    st.write("ANOVA Result", anova_result)

    tukey = pairwise_tukeyhsd(endog=df["Burnout_Level_Encoded"], groups=df["Work_Arrangement"], alpha=0.05)
    st.text("Tukey HSD Result")
    st.text(tukey.summary())

# ---------- TAB 3: Predictive Model ----------
with tab3:
    st.header("🤖 Random Forest Burnout Prediction")
    X = pd.get_dummies(df[features], drop_first=True)
    y = df["Burnout_Level_Encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values("Importance", ascending=False)
    fig = px.bar(feat_df.head(10), x="Importance", y="Feature", orientation="h", title="Top 10 Feature Importances")
    st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 4: Scenario Simulation ----------
with tab4:
    st.header("🔮 Simulate Burnout Risk by Work Mode")
    persona = df[(df["Job_Role"] == selected_job) & (df["Region"] == selected_region)].iloc[0]

    scenarios = []
    for mode in selected_work_modes:
        sc = persona.copy()
        sc["Work_Arrangement"] = mode
        scenarios.append(sc)
    sc_df = pd.DataFrame(scenarios)

    sc_X = pd.get_dummies(sc_df[features], drop_first=True)
    sc_X = sc_X.reindex(columns=X.columns, fill_value=0)
    sc_df["Predicted_Burnout"] = model.predict(sc_X)
    sc_df["Label"] = sc_df["Predicted_Burnout"].map({0: "Low", 1: "Medium", 2: "High"})

    st.dataframe(sc_df[["Work_Arrangement", "Predicted_Burnout", "Label"]])

    # SHAP explainability
    st.subheader("🔍 SHAP Waterfall (First Scenario)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sc_X)

    fig, ax = plt.subplots()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value[1],
        shap_values[1][0],
        feature_names=sc_X.columns,
        max_display=10,
        show=False
    )
    st.pyplot(fig)

    st.caption("Waterfall plot explains why the model gave this burnout prediction.")

# ---------- END ----------
st.markdown("---")
st.markdown("🧠 Built by Senior Andiswa Mabuza - Amabuza53@gmail.com | © 2025")
