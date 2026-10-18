import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="NovaGen Health Risk ML", page_icon="🧬", layout="wide")

st.title("🧬 NovaGen Research Labs")
st.subheader("Supervised Machine Learning — Health Outcome Classification")
st.caption("Assignment 5 | Educational ML demonstration")

DATA_PATH = "novagen_dataset.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    if "Target" not in data.columns:
        raise ValueError("Dataset must contain a 'Target' column.")
    data = data.dropna(subset=["Target"]).copy()
    data["Target"] = data["Target"].astype(int)
    return data

@st.cache_resource
def train_models(data):
    X = data.drop(columns=["Target"])
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    fitted, results = {}, []
    for name, estimator in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
        })
    metrics = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    best_name = metrics.loc[0, "Model"]
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)
    return X, y, X_train, X_test, y_train, y_test, fitted, metrics, best_name, best_model, best_pred

try:
    data = load_data()
    X, y, X_train, X_test, y_train, y_test, models, metrics, best_name, best_model, best_pred = train_models(data)
except Exception as exc:
    st.error(f"Could not load/train the dataset: {exc}")
    st.stop()

with st.sidebar:
    st.header("Project menu")
    page = st.radio("Go to", ["Overview", "Predict an example", "Model evaluation", "Dataset explorer"])
    st.divider()
    st.caption(f"Dataset rows: {len(data):,}")
    st.caption(f"Input features: {X.shape[1]}")

if page == "Overview":
    st.markdown("""
    This app trains and compares three supervised classification algorithms using the supplied assignment dataset.
    The target column is **Target**. The dataset description says it represents a health outcome/risk label.
    """)
    a, b, c = st.columns(3)
    a.metric("Records", f"{len(data):,}")
    b.metric("Input features", X.shape[1])
    c.metric("Classes", y.nunique())
    st.markdown("### Target distribution")
    counts = data["Target"].value_counts().sort_index()
    target_df = pd.DataFrame({"Target class": counts.index.astype(str), "Records": counts.values})
    st.bar_chart(target_df.set_index("Target class"))
    st.markdown("### Model comparison")
    st.dataframe(metrics.style.format({col: "{:.3f}" for col in ["Accuracy", "Precision", "Recall", "F1 Score"]}), use_container_width=True)
    st.info("Interpret Target=0 and Target=1 only according to your instructor's dataset definition. This is an educational assignment, not a medical diagnosis or clinical decision tool.")

elif page == "Predict an example":
    st.markdown("Enter values for the dataset's feature columns. Coded categories are kept as numeric codes because the assignment sheet does not define every code.")
    st.warning("This output is a model estimate based on the provided assignment data, not medical advice. Do not use it to make real health decisions.")
    with st.form("prediction_form"):
        inputs = {}
        left, right = st.columns(2)
        for i, col in enumerate(X.columns):
            box = left if i % 2 == 0 else right
            vals = X[col]
            if pd.api.types.is_bool_dtype(vals):
                inputs[col] = box.selectbox(col, [False, True], format_func=lambda v: "Yes" if v else "No")
            elif pd.api.types.is_numeric_dtype(vals):
                lo = float(vals.min())
                hi = float(vals.max())
                med = float(vals.median())
                step = 1.0 if pd.api.types.is_integer_dtype(vals) or col in [
                    "Age", "Blood_Pressure", "Cholesterol", "Glucose_Level", "Heart_Rate",
                    "Stress_Level", "Smoking", "Alcohol", "Diet", "MentalHealth",
                    "PhysicalActivity", "MedicalHistory", "Allergies"
                ] else 0.1
                inputs[col] = box.number_input(col, min_value=lo, max_value=hi, value=med, step=step)
            else:
                options = sorted(vals.dropna().unique().tolist())
                inputs[col] = box.selectbox(col, options)
        submitted = st.form_submit_button("Predict target class", type="primary")
    if submitted:
        sample = pd.DataFrame([inputs], columns=X.columns)
        prediction = int(best_model.predict(sample)[0])
        st.subheader("Prediction result")
        st.metric("Predicted Target class", str(prediction))
        if hasattr(best_model, "predict_proba"):
            probabilities = best_model.predict_proba(sample)[0]
            probability_df = pd.DataFrame({"Target class": [str(c) for c in best_model.classes_], "Model probability": probabilities})
            st.bar_chart(probability_df.set_index("Target class"))
        st.caption(f"Model selected by test-set F1 score: {best_name}. Interpret the class label using your assignment instructions.")

elif page == "Model evaluation":
    st.markdown("### Holdout test-set metrics")
    st.caption("An 80/20 stratified train/test split is used. These are measured on the held-out test set.")
    st.dataframe(metrics.style.format({col: "{:.4f}" for col in ["Accuracy", "Precision", "Recall", "F1 Score"]}), use_container_width=True)
    st.markdown(f"### Best model by F1 score: {best_name}")
    cm = confusion_matrix(y_test, best_pred, labels=best_model.classes_)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(best_model.classes_)), [str(v) for v in best_model.classes_])
    ax.set_yticks(range(len(best_model.classes_)), [str(v) for v in best_model.classes_])
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)
    st.markdown("### Classification report")
    st.code(classification_report(y_test, best_pred, zero_division=0))
    model = models[best_name].named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        st.markdown("### Top feature importances")
        st.bar_chart(importances)

elif page == "Dataset explorer":
    st.markdown("### Preview")
    st.dataframe(data.head(100), use_container_width=True)
    st.markdown("### Data quality")
    quality = pd.DataFrame({
        "Column": data.columns,
        "Data type": [str(t) for t in data.dtypes],
        "Missing values": [int(v) for v in data.isna().sum().values],
        "Unique values": [int(v) for v in data.nunique().values],
    })
    st.dataframe(quality, use_container_width=True)
    st.markdown("### Summary statistics")
    st.dataframe(data.describe(include="all").transpose(), use_container_width=True)
    st.download_button("Download dataset CSV", data=data.to_csv(index=False).encode("utf-8"), file_name="novagen_dataset.csv", mime="text/csv")

st.divider()
st.caption("NovaGen Research Labs | Supervised ML Assignment 5 | For learning purposes only")
