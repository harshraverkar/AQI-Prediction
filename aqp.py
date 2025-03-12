import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import seaborn as sns

# Load the dataset
data = pd.read_csv("C://Users//harsh//OneDrive//Desktop//vs code//AQI//concatenated_file.csv")

# Preprocessing and Feature Engineering
data.fillna(data.median(numeric_only=True), inplace=True)  # Fill missing numeric values with median
data['CO_to_NO2'] = data['CO AQI Value'] / (data['NO2 AQI Value'] + 1)  # Avoid division by zero
data['Ozone_to_PM25'] = data['Ozone AQI Value'] / (data['PM2.5 AQI Value'] + 1)

# Define feature and target columns
feature_columns = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'CO_to_NO2', 'Ozone_to_PM25']
target_column = 'PM2.5 AQI Value'

# Split the data
X = data[feature_columns]
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to calculate AQI level
def get_aqi_level(pm25_aqi):
    if 0 <= pm25_aqi <= 50: return "Good"
    elif 51 <= pm25_aqi <= 100: return "Moderate"
    elif 101 <= pm25_aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif 151 <= pm25_aqi <= 200: return "Unhealthy"
    elif 201 <= pm25_aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

# Main Application Class
class AirQualityApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Air Quality Predictor")
        self.geometry("1200x800")
        self.configure(bg="#E6F0FA")

        # Center the window
        self.update_idletasks()
        width, height = 1200, 800
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

        # Container
        self.container = tk.Frame(self, bg="#E6F0FA")
        self.container.pack(expand=True, fill="both", padx=20, pady=20)

        self.frames = {}
        for F in (InputPage, ResultsPage):
            frame = F(self.container, self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.show_frame("InputPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

# Input Page
class InputPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#E6F0FA")
        self.controller = controller

        # Background image
        window_width, window_height = self.winfo_screenwidth(), self.winfo_screenheight()
        image = Image.open("aqi.png")
        resized_image = image.resize((window_width, window_height), Image.Resampling.LANCZOS)
        self.background_image = ImageTk.PhotoImage(resized_image)
        background_label = tk.Label(self, image=self.background_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        background_label.lower()

        # Styling
        style = ttk.Style()
        style.configure("TButton", padding=15, font=("Helvetica", 14, "bold"), background="#4CAF50", foreground="white")
        style.map("TButton", background=[("active", "#45A049")])
        style.configure("TLabel", font=("Helvetica", 16), background="#E6F0FA", foreground="#1E3A8A")

        content_frame = tk.Frame(self, bg="#E6F0FA")
        content_frame.pack(expand=True)

        # Larger Logo
        try:
            logo = Image.open("logo.png")
            logo_resized = logo.resize((300, 300), Image.Resampling.LANCZOS)  # Increased size
            self.logo_image = ImageTk.PhotoImage(logo_resized)
            logo_label = tk.Label(content_frame, image=self.logo_image, bg="#E6F0FA")
            logo_label.pack(pady=40)
        except:
            pass

        # Input fields
        co_label = ttk.Label(content_frame, text="CO AQI Value")
        co_label.pack(pady=10)
        self.co_entry = tk.Entry(content_frame, font=("Helvetica", 20), fg="#1E3A8A", bg="#FFFFFF", width=30)
        self.co_entry.pack(pady=10)

        ozone_label = ttk.Label(content_frame, text="Ozone AQI Value")
        ozone_label.pack(pady=10)
        self.ozone_entry = tk.Entry(content_frame, font=("Helvetica", 20), fg="#1E3A8A", bg="#FFFFFF", width=30)
        self.ozone_entry.pack(pady=10)

        no2_label = ttk.Label(content_frame, text="NO2 AQI Value")
        no2_label.pack(pady=10)
        self.no2_entry = tk.Entry(content_frame, font=("Helvetica", 20), fg="#1E3A8A", bg="#FFFFFF", width=30)
        self.no2_entry.pack(pady=10)

        predict_button = ttk.Button(content_frame, text="Predict and Analyze", command=self.predict_and_switch, style="TButton")
        predict_button.pack(pady=30)

    def predict_and_switch(self):
        try:
            co_value = float(self.co_entry.get())
            ozone_value = float(self.ozone_entry.get())
            no2_value = float(self.no2_entry.get())

            # Feature engineering for prediction
            co_to_no2 = co_value / (no2_value + 1)
            ozone_to_pm25 = ozone_value / (np.mean(data['PM2.5 AQI Value']) + 1)  # Approximation
            new_data = pd.DataFrame({'CO AQI Value': [co_value], 'Ozone AQI Value': [ozone_value], 
                                     'NO2 AQI Value': [no2_value], 'CO_to_NO2': [co_to_no2], 
                                     'Ozone_to_PM25': [ozone_to_pm25]})
            
            predicted_pm25_aqi = model.predict(new_data)[0]
            aqi_level = get_aqi_level(predicted_pm25_aqi)

            self.controller.frames["ResultsPage"].update_results(predicted_pm25_aqi, aqi_level, 
                                                                 co_value, ozone_value, no2_value)
            self.controller.show_frame("ResultsPage")
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")

# Results Page
class ResultsPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#E6F0FA")
        self.controller = controller

        # Background image
        window_width, window_height = self.winfo_screenwidth(), self.winfo_screenheight()
        image = Image.open("aqi.png")
        resized_image = image.resize((window_width, window_height), Image.Resampling.LANCZOS)
        self.background_image = ImageTk.PhotoImage(resized_image)
        background_label = tk.Label(self, image=self.background_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        background_label.lower()

        # Styling
        style = ttk.Style()
        style.configure("TButton", padding=15, font=("Helvetica", 14, "bold"), background="#4CAF50", foreground="white")
        style.map("TButton", background=[("active", "#45A049")])
        style.configure("TLabel", font=("Helvetica", 16), background="#E6F0FA", foreground="#1E3A8A")

        # Layout: Left (graphs) and Right (results)
        self.left_frame = tk.Frame(self, bg="#E6F0FA", width=600)
        self.left_frame.pack(side=tk.LEFT, fill="y", padx=10, pady=10)

        self.right_frame = tk.Frame(self, bg="#E6F0FA", width=600)
        self.right_frame.pack(side=tk.RIGHT, fill="both", expand=True, padx=10, pady=10)

        # Left frame: Three graph sections
        self.bar_frame = tk.Frame(self.left_frame, bg="#FFFFFF", bd=2, relief="sunken")
        self.bar_frame.pack(fill="x", expand=True, pady=5)

        self.line_frame = tk.Frame(self.left_frame, bg="#FFFFFF", bd=2, relief="sunken")
        self.line_frame.pack(fill="x", expand=True, pady=5)

        self.scatter_frame = tk.Frame(self.left_frame, bg="#FFFFFF", bd=2, relief="sunken")
        self.scatter_frame.pack(fill="x", expand=True, pady=5)

        # Right frame: Results and EDA
        self.aqi_label = ttk.Label(self.right_frame, text="")
        self.aqi_label.pack(pady=10)

        self.locations_label = ttk.Label(self.right_frame, text="", wraplength=500, justify="left")
        self.locations_label.pack(pady=10)

        self.eda_frame = tk.Frame(self.right_frame, bg="#FFFFFF", bd=2, relief="sunken")
        self.eda_frame.pack(fill="both", expand=True, pady=10)

        back_button = ttk.Button(self.right_frame, text="Back", 
                                 command=lambda: controller.show_frame("InputPage"), style="TButton")
        back_button.pack(pady=20)

        # Placeholders for canvases
        self.bar_canvas = self.line_canvas = self.scatter_canvas = self.eda_canvas = None

    def update_results(self, predicted_pm25_aqi, aqi_level, co_value, ozone_value, no2_value):
        # Update AQI and locations
        self.aqi_label.config(text=f"Predicted PM2.5 AQI: {predicted_pm25_aqi:.2f} ({aqi_level})")
        
        closest_locations = data.iloc[(data[target_column] - predicted_pm25_aqi).abs().argsort()[:10]]
        locations_text = "10 Closest Locations by PM2.5 AQI:\n"
        for idx, row in closest_locations.iterrows():
            locations_text += f"{row['City']}, {row['Country']} - PM2.5 AQI: {row[target_column]}\n"
        self.locations_label.config(text=locations_text)

        # Store values
        self.co_value, self.ozone_value, self.no2_value = co_value, ozone_value, no2_value
        self.predicted_pm25_aqi = predicted_pm25_aqi

        # Generate graphs and EDA
        self.generate_graphs()
        self.generate_eda()

    def generate_graphs(self):
        new_data = pd.DataFrame({'CO AQI Value': [self.co_value], 'Ozone AQI Value': [self.ozone_value], 
                                 'NO2 AQI Value': [self.no2_value]})

        # Clear previous graphs
        for canvas in [self.bar_canvas, self.line_canvas, self.scatter_canvas]:
            if canvas: canvas.get_tk_widget().destroy()

        # Bar Graph
        fig_bar, ax_bar = plt.subplots(figsize=(5, 2))
        new_data.plot(kind='bar', ax=ax_bar, legend=False, color=['#1E3A8A', '#4CAF50', '#FF6F61'])
        ax_bar.set_title('Input Values (Bar)', fontsize=10)
        self.bar_canvas = FigureCanvasTkAgg(fig_bar, master=self.bar_frame)
        self.bar_canvas.draw()
        self.bar_canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig_bar)

        # Line Graph
        fig_line, ax_line = plt.subplots(figsize=(5, 2))
        new_data.plot(kind='line', ax=ax_line, legend=False, marker='o', color=['#1E3A8A', '#4CAF50', '#FF6F61'])
        ax_line.set_title('Input Values (Line)', fontsize=10)
        self.line_canvas = FigureCanvasTkAgg(fig_line, master=self.line_frame)
        self.line_canvas.draw()
        self.line_canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig_line)

        # Scatter Plot
        fig_scatter, ax_scatter = plt.subplots(figsize=(5, 2))
        ax_scatter.scatter(data['CO AQI Value'], data['PM2.5 AQI Value'], label='CO', color='#1E3A8A', alpha=0.5)
        ax_scatter.scatter(data['Ozone AQI Value'], data['PM2.5 AQI Value'], label='Ozone', color='#4CAF50', alpha=0.5)
        ax_scatter.scatter(data['NO2 AQI Value'], data['PM2.5 AQI Value'], label='NO2', color='#FF6F61', alpha=0.5)
        ax_scatter.plot([self.co_value, self.ozone_value, self.no2_value], [self.predicted_pm25_aqi]*3, 'ro', label='Predicted')
        ax_scatter.set_title('AQI vs PM2.5 (Scatter)', fontsize=10)
        ax_scatter.legend(fontsize=8)
        self.scatter_canvas = FigureCanvasTkAgg(fig_scatter, master=self.scatter_frame)
        self.scatter_canvas.draw()
        self.scatter_canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig_scatter)

    def generate_eda(self):
        if self.eda_canvas: self.eda_canvas.get_tk_widget().destroy()

        # Correlation Heatmap
        fig, ax = plt.subplots(figsize=(5, 4))
        corr = data[feature_columns + [target_column]].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        ax.set_title("Correlation Heatmap", fontsize=12)
        self.eda_canvas = FigureCanvasTkAgg(fig, master=self.eda_frame)
        self.eda_canvas.draw()
        self.eda_canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

        # Additional EDA (e.g., Top Polluted Cities) could be added in a separate frame or tab

# Run the application
if __name__ == "__main__":
    app = AirQualityApp()
    app.mainloop()