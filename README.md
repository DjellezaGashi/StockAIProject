# StockAIProject

## 🧠 Overview
**StockAIProject** is an AI-driven experiment that predicts whether a stock’s
closing price will **increase or decrease** the next day based on historical
market data.  

All of the design, feature engineering, and model-building steps were created
entirely through **AI prompting** — this project demonstrates how effective
AI can be in guiding the end-to-end process of building machine learning
systems.

---

## ⚙️ Features
- Uses **real stock data** collected from Yahoo Finance (`yfinance`)
- Engineers realistic **technical indicators** such as:
  - Daily Return  
  - Price Range  
  - Moving Averages (MA)  
  - Volatility  
- Trains and compares two machine learning models:
  - Logistic Regression (baseline)
  - Random Forest Classifier (non-linear pattern learning)
- Evaluates results using accuracy, precision, recall, and confusion matrix
- Visualizes feature correlations with **heatmaps**

---

## 📊 Results
The Random Forest model achieves an average accuracy of **~55%**, which aligns
with real-world expectations for short-term stock movement prediction.
Even professional trading models rarely exceed 60% accuracy — making this a
strong baseline for educational purposes.

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/StockAIProject.git
   cd StockAIProject
2. Install dependencies
    ```bash
    pip install -r requirements.txt
3. Run the data collection script:
    ```bash
    python3 stockData.py
4. Train and evaluate the AI model:
    ```bash
    python3 stockModel.py


## Disclaimer
This project is for educational purposes only.
It does not provide financial advice or guarantee trading performance

## 🪄 Credits
Created by **Djelleza Gashi** 
All design, code, and documentation were developed collaboratively with AI through iterative prompt engineering.

