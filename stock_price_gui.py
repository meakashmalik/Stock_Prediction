import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import os

MODEL_PATH = 'linear_stock_model.pkl'

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    from sklearn.linear_model import LinearRegression
    days = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
    prices = np.array([150,152,153,155,157,160,162,165,168,170])
    model = LinearRegression().fit(days, prices)

if os.path.exists('sample_stock.csv'):
    df = pd.read_csv('sample_stock.csv')
    plot_days = df['Day'].values.reshape(-1,1)
    plot_prices = df['Close'].values
else:
    plot_days = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
    plot_prices = np.array([150,152,153,155,157,160,162,165,168,170])

root = tk.Tk()
root.title("📈 Stock Price Predictor")
root.geometry("460x340")
root.config(bg="#f8f8f8")

tk.Label(root, text="Stock Price Prediction", font=("Helvetica",16,"bold"), bg="#f8f8f8").pack(pady=10)
tk.Label(root, text="Enter Day Number:", font=("Helvetica",12), bg="#f8f8f8").pack(pady=6)
day_entry = tk.Entry(root, font=("Helvetica",12), width=12)
day_entry.pack()

def predict_price():
    try:
        day = float(day_entry.get())
        if day <= 0:
            messagebox.showerror("Error","Enter positive day number!")
            return
        predicted = model.predict([[day]])[0]
        result_label.config(text=f"Predicted Price: ₹{predicted:.2f}", fg="green")
    except Exception:
        messagebox.showerror("Error","Enter a valid number!")

def show_graph():
    plt.figure(figsize=(8,5))
    plt.scatter(plot_days, plot_prices, label='Actual Data')
    xline = np.arange(min(plot_days.flatten()), max(plot_days.flatten())+10).reshape(-1,1)
    plt.plot(xline, model.predict(xline), label='Regression Line', color='red')
    plt.title('Stock Price Prediction')
    plt.xlabel('Day')
    plt.ylabel('Price (₹)')
    plt.legend()
    plt.show()

tk.Button(root, text="Predict Price", command=predict_price, font=("Helvetica",12), bg="#4CAF50", fg="white").pack(pady=10)
tk.Button(root, text="Show Graph", command=show_graph, font=("Helvetica",12), bg="#2196F3", fg="white").pack()
result_label = tk.Label(root, text="", font=("Helvetica",14,"bold"), bg="#f8f8f8")
result_label.pack(pady=14)
root.mainloop()
