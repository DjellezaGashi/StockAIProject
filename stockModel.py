# -------------------------------------------------------
# AI Stock Movement Classifier (Upgraded Feature Version)
# -------------------------------------------------------

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. LOAD AND CLEAN THE DATA
# -------------------------------------------------------
df = pd.read_csv("real_stocks.csv")

# Drop irrelevant or missing columns
df = df.drop(columns=["Adj Close"], errors="ignore")
df = df.dropna()

# -------------------------------------------------------
# 1B. FEATURE ENGINEERING (SMART FEATURES)
# -------------------------------------------------------
df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]
df["Price_Range"] = (df["High"] - df["Low"]) / df["Open"]
df["MA_5"] = df["Close"].rolling(window=5).mean()
df["Volatility_5"] = df["Close"].rolling(window=5).std()

# Drop rows with NaN values caused by rolling windows
df = df.dropna()

print("\nAdded new engineered features. Rows after rolling windows:", len(df))
print(df[["Date", "Close", "Daily_Return", "Price_Range", "MA_5", "Volatility_5"]].head())

# -------------------------------------------------------
# 1C. ADD MORE TECHNICAL INDICATORS
# -------------------------------------------------------

# Momentum: price difference over 5 days
df["Momentum_5"] = df["Close"] - df["Close"].shift(5)

# Moving averages
df["SMA_10"] = df["Close"].rolling(window=10).mean()
df["SMA_20"] = df["Close"].rolling(window=20).mean()

# Moving average crossover (trend indicator)
df["MA_Cross"] = df["SMA_10"] - df["SMA_20"]

# RSI (Relative Strength Index) 14-day
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# Drop rows with NaN values from rolling windows
df = df.dropna()


# -------------------------------------------------------
# 2. CORRELATION CHECK (Visual)
# -------------------------------------------------------
corr = df[["Daily_Return", "Price_Range", "MA_5", "Volatility_5"]].corr()
print("\nFeature Correlations:\n", corr)

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap (Engineered Features)")
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 3. FEATURE SELECTION
features = [
    "Daily_Return", "Price_Range", "Volatility_5",
    "Momentum_5", "SMA_10", "SMA_20", "MA_Cross", "RSI_14"
]

target = "UpDown"

X = df[features]
y = df[target]

# -------------------------------------------------------
# 4. TRAIN/TEST SPLIT
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -------------------------------------------------------
# 5. SCALING
# -------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------
# 6. TRAIN MODELS (CLASS BALANCED)
# -------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# -------------------------------------------------------
# 7. EVALUATION
# -------------------------------------------------------
print("\n=== Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log, zero_division=0))

print("\n=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, zero_division=0))

print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))
