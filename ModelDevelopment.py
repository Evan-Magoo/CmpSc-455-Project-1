import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

# Load Preprocessed DataSet CSVs
scrape_df = pd.read_csv("scrape_data_processed.csv")
api_df = pd.read_csv("api_data_processed.csv")
match_df = pd.read_csv("match_data_processed.csv")

# Encode Category Labels For Classification and Labeling
encoder = LabelEncoder()
temp_df = api_df.copy()
temp_df['category_encoded'] = encoder.fit_transform(temp_df['category_label'])
label_map = dict(enumerate(encoder.classes_))

# For Comparing Data
results = []

# Removes matching data from another dataframe
def remove_overlap(train_df, test_df):
    overlap_ids = set(train_df['video_id']).intersection(test_df['video_id'])
    train_df = train_df[~train_df['video_id'].isin(overlap_ids)].copy()
    return train_df

# XGBoost Regression Model
def train_xgb(df, model_name, target):
    print(f"\n\tXGBoost Regression Model - Targeting: {target}\n")
    dropped_features = [target, 'target_log', 'video_id', 'upload_day', 'category_label']
    df = remove_overlap(df, match_df)
    # -----------------
    clip_val = df[target].quantile(0.99)
    df[target] = np.clip(df[target], 0, clip_val)

    match_clip_val = match_df[target].quantile(0.99)
    match_df_clipped = match_df.copy()
    match_df_clipped[target] = np.clip(match_df_clipped[target], 0, match_clip_val)

    df['target_log'] = np.log1p(df[target])
    match_df_clipped['target_log'] = np.log1p(match_df_clipped[target])

    y_train = df['target_log']
    y_test = match_df_clipped['target_log']

    # -----------------
    X_train = df.drop(columns=dropped_features)
    X_train = X_train.select_dtypes(include=[np.number])

    X_test = match_df_clipped.drop(columns=dropped_features)
    X_test = X_test.select_dtypes(include=[np.number])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


    # ----------------- Model
    xg_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=1.0,
        learning_rate=0.01,
        max_depth=5,
        n_estimators=500,
        subsample=0.7,
        random_state=42
    )

    # ----------------- RFE w/ Cross-Validation
    print("\t---- Feature Selection -----------------------------------------")
    rfe = RFECV(
        estimator=xg_reg,
        step=1,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    rfe.fit(X_train, y_train)

    print("\tOptimal number of features:", rfe.n_features_)
    top_features = X_train.columns[rfe.support_]
    print("\tTop features:", list(top_features))

    X_train_rfe = X_train[top_features]
    X_test_rfe = X_test[top_features]

    # ----------------- Train and Predict -----------------
    xg_reg.fit(X_train_rfe, y_train)
    y_pred_test = xg_reg.predict(X_test_rfe)
    y_pred_train = xg_reg.predict(X_train_rfe)

    # ----------------- Metrics -----------------
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    mset = mean_squared_error(y_train, y_pred_train)
    r2_t = r2_score(y_train, y_pred_train)
    print("\n\t---- Metrics ---------------------------------------------------")
    print(f"\tMSE (Train) = {mset:.6f}")
    print(f"\tR2 (Train) = {r2_t:.4f}")
    print(f"\n\tMSE (Test) = {mse:.6f}")
    print(f"\tR2 (Test) = {r2:.4f}")

    # ----------------- Visualization -----------------
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred_test, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()],[(y_test.min()), y_test.max()], 'r--')
    x_label = "Actual " + target + " (log)"
    plt.xlabel(x_label)
    y_label = "Predicted " + target + " (log)"
    plt.ylabel(y_label)
    title = model_name + " - Actual vs Predicted (XGBoost)"
    plt.title(title)
    plt.show()
    plot_xgb_feature_importance(xg_reg, top_features, model_name)
    print()

    # Save Performance
    results.append({
        "model": "XGBoost",
        "source": model_name,
        "train_score": r2_t,
        "test_score": r2
    })

# Logistic Regression Model
def train_log(df, model_name, target):

    print(f"\n\tLogistical Regression Model - Targeting: {target}\n")
    df = df.copy()  # avoid modifying original df
    df = remove_overlap(df, match_df)


    y_train = df[target]
    y_test = match_df[target]

    # ----------------- Features
    dropped_features = [target, 'video_id', 'upload_day', 'category_label']
    X_train = df.drop(columns=dropped_features + [target]).select_dtypes(include=[np.number])
    X_test = match_df.drop(columns=dropped_features + [target]).select_dtypes(include=[np.number])

    # ----------------- Align test features with train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # ----------------- Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------- Model
    model = LogisticRegression(
        class_weight='balanced',
        penalty='l2',
        solver='lbfgs',
        C=1000,
        max_iter=2500,
        random_state=42
    )

    # ----------------- RFE w/ Cross-Validation
    print("\t---- Feature Selection -----------------------------------------")

    rfe = RFECV(
        estimator=model,
        step=1,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    rfe.fit(X_train_scaled, y_train)

    print("\tOptimal number of features:", rfe.n_features_)
    top_features = X_train.columns[rfe.support_]
    print("\tTop features:", list(top_features))

    X_train_rfe = X_train_scaled[:, rfe.support_]
    X_test_rfe = X_test_scaled[:, rfe.support_]

    # ----------------- Train best model
    model.fit(X_train_rfe, y_train)
    y_pred_test = model.predict(X_test_rfe)
    y_pred_train = model.predict(X_train_rfe)

    # ----------------- Metrics
    accuracy = accuracy_score(y_test, y_pred_test)
    accuracy_t = accuracy_score(y_train, y_pred_train)

    print("\n\t---- Metrics ---------------------------------------------------")
    print(f"\tAccuracy (Test) = {accuracy:.4f}")
    print(f"\tAccuracy (Train) = {accuracy_t:.4f}")

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    plt.figure(figsize=(8,7))
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Predicted Categories')
    plt.ylabel('True Categories')
    plt.title(f"{model_name} - Confusion Matrix (All Categories)")
    plt.subplots_adjust(
        left=0.25,
        right=1.0,
        bottom=0.2,
        top=0.95
    )
    plt.show()

    # Feature Important Plot
    plot_feature_importance(model, top_features, model_name)

    # Save Results
    results.append({
        "model": "Logistic Regression",
        "source": model_name,
        "train_score": accuracy_t,
        "test_score": accuracy
    })

# XGBoost Regression Visualization - Top 20 Important Features
def plot_xgb_feature_importance(xgb_model, feature_names, model_name):
    # ----------------- Get Coefficients
    importance = xgb_model.get_booster().get_score(importance_type='gain')
    importance_df = (
        pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })
        .sort_values(by='Importance', ascending=False)
    )

    # ----------------- Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][:20][::-1], importance_df['Importance'][:20][::-1])
    plt.xlabel('Gain')
    plt.title(f'{model_name} — Top 20 XGBoost Feature')
    plt.tight_layout()
    plt.show()

# Logistic Regression Visualization - Top 20 Important Features
def plot_feature_importance(model, feature_names, model_name):
    # ----------------- Get Coefficients
    if model.coef_.ndim > 1:  # multiclass
        importance = np.mean(np.abs(model.coef_), axis=0)
    else:
        importance = np.abs(model.coef_[0])

    # ----------------- Create Dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    # ----------------- Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][:20][::-1], importance_df['Importance'][:20][::-1])
    plt.xlabel('Absolute Coefficient')
    plt.title(f"{model_name} — Top 20 Logistic Regression Features")
    plt.tight_layout()
    plt.show()

# Visualize Average Engagement (Likes/Views) per Category
def engagement_visualization(df, model_name):
    df = df.copy()
    df['engagement'] = df['likes'] / df['views']
    category_stats = (df.groupby('category')['engagement'].mean().sort_values(ascending=False))

    plt.figure(figsize=(12, 6))
    sns.barplot(x=encoder.classes_, y=category_stats.values)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"{model_name} - Engagement by Category")
    plt.xlabel("Category")
    plt.ylabel("Engagement")
    plt.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.3,
        top=0.9
    )
    plt.show()

# Compare Performance of Models Trained on Scraped vs API data
def compare_models():
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Plot test performance comparison between Scraped vs API
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=results_df,
        x="model",
        y="test_score",
        hue="source",
        palette="Set2"
    )
    plt.title("Comparison of Model Performance: Scraped vs API Data")
    plt.ylabel("Performance (R² for XGBoost, Accuracy for Logistic Regression)")
    plt.xlabel("Model Type")
    plt.ylim(0, 1)
    plt.legend(title="Data Source")
    plt.tight_layout()
    plt.show()

# Main Method
if __name__ == "__main__":
    results = []

    print("\nWeb Scraping Models")
    print("--------------------------------------------------------------------")
    print("\tData Amount:", len(scrape_df), "entries")

    # Scraped Data Training
    train_xgb(scrape_df, "Scrape",'views')
    train_log(scrape_df, "Scrape", 'category')
    engagement_visualization(scrape_df, "Scrape")

    print("\nYouTube API Models")
    print("--------------------------------------------------------------------")
    print("\tData Amount:", len(api_df), "entries")

    # API Data Training
    train_xgb(api_df, "API", 'views')
    train_log(api_df, "API", 'category')
    engagement_visualization(api_df, "API")
    print()

    compare_models()

