"""
Diabetes Management Application - Interactive GUI Interface

This application provides a user-friendly interface for diabetes management and risk assessment.
It uses machine learning (XGBoost) to predict diabetes risk scores based on daily health metrics.

Features:
- User Profile Management:
  * User ID and basic information collection
  * Real-time input validation
  * Interactive form with tooltips

- Daily Health Tracking:
  * Blood Glucose (70-300 mg/dL)
  * Physical Activity (minutes/day)
  * Diet (healthy/unhealthy)
  * Medication Adherence
  * Stress Levels
  * Sleep Hours
  * Hydration Status

- Risk Assessment:
  * Real-time risk calculation using XGBoost model
  * Risk categories with emoji feedback:
    - Low Risk (<30): 🎉 Celebration animation
    - Moderate Risk (30-60): ⚠️ Warning feedback
    - High Risk (>60): 🚨 Urgent attention needed

- UI Features:
  * Animated text and emojis
  * Color-changing buttons
  * Interactive tooltips
  * Progress indicators
  * Motivational quotes

- Data Management:
  * FHIR-compatible data formatting
  * Simulated EHR integration
  * JSON data exchange format

Dependencies:
- tkinter: GUI framework
- numpy: Numerical computations
- joblib: Model loading
- json: Data formatting
- datetime: Timestamp management

Usage:
    python app.py

Note: Requires pre-trained model (diabetes_risk_model.pkl) and scaler (scaler.pkl)

Authors: Kevin Tan, Haichao Min, Hanfu Hou, Shreyas Karnad, You Wu, Donald Su
Date: 2024
"""
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import font as tkFont
from datetime import datetime
import random
import json
import numpy as np
import joblib
import time

# Load the trained diabetes risk model and scaler
xgb_model = joblib.load('diabetes_risk_model.pkl')
scaler = joblib.load('scaler.pkl')

class DiabetesManagementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Management App 😊")
        self.root.geometry("600x800")
        self.root.configure(bg="black")  # Black background for better contrast

        # Set a global style for fonts
        self.label_font = tkFont.Font(family="Helvetica", size=16, weight="bold")
        self.entry_font = tkFont.Font(family="Helvetica", size=16)
        self.header_font = tkFont.Font(family="Helvetica", size=22, weight="bold")
        self.button_font = tkFont.Font(family="Helvetica", size=18, weight="bold")
        self.tip_font = tkFont.Font(family="Helvetica", size=12, weight="bold", slant="italic")

        # User Information
        self.user_id = tk.StringVar()
        self.user_name = tk.StringVar()
        self.age = tk.StringVar()
        self.weight = tk.StringVar()
        self.height = tk.StringVar()
        self.activity_level = tk.StringVar()

        # Daily Data
        self.blood_glucose = tk.StringVar()
        self.diet = tk.StringVar()
        self.physical_activity = tk.StringVar()
        self.medication_adherence = tk.StringVar()
        self.stress_level = tk.StringVar()
        self.sleep_hours = tk.StringVar()
        self.hydration_level = tk.StringVar()

        # Add animation colors
        self.colors = ["#1abc9c", "#2ecc71", "#3498db", "#9b59b6", "#f1c40f"]
        self.current_color_index = 0

        # Start color animation
        self.animate_colors()

        self.bmi = tk.StringVar()  # Add BMI variable

        # Define advice dictionary for health metrics
        self.advice = {
            "Physical Activity": {
                "low": "⚡ Less than 30 minutes of activity. Aim for at least 30 minutes of moderate exercise daily.",
                "moderate": "👍 Good activity level! Try to maintain or gradually increase your activity.",
                "high": "🌟 Excellent activity level! Remember to stay hydrated and rest adequately."
            },
            
            "BMI": {
                "Underweight": "🏋️ Consider increasing caloric intake and strength training exercises.",
                "Normal weight": "✨ Great job maintaining a healthy weight! Keep up your current habits.",
                "Overweight": "🚶 Consider increasing physical activity and watching portion sizes.",
                "Obese": "⚕️ Please consult with a healthcare provider about a weight management plan."
            },
            
            "Blood Glucose": {
                "low": "🍎 Blood glucose is low. Consider regular small meals and consulting your doctor.",
                "normal": "✅ Blood glucose is in a healthy range. Maintain your current diet and medication routine.",
                "high": "⚠️ Blood glucose is elevated. Review your diet and medication adherence.",
                "extreme": "🚨 Blood glucose level is critically high. Please seek medical attention immediately!"
            },
            
            "Diet": {
                "healthy": "🥗 Excellent dietary choices! Keep maintaining a balanced, healthy diet.",
                "unhealthy": "🍏 Consider incorporating more whole foods and reducing processed foods."
            },
            
            "Medication": {
                "good": "💊 Great medication adherence! Keep maintaining this routine.",
                "poor": "⚕️ Important to take medications as prescribed. Set reminders if needed."
            },
            
            "Stress": {
                "low": "😊 Great stress management! Keep using your effective coping strategies.",
                "medium": "🧘 Consider stress-reduction techniques like meditation or yoga.",
                "high": "⚡ High stress detected. Please prioritize stress management and consider professional support."
            },
            
            "Sleep": {
                "low": "😴 Less than 7 hours of sleep. Aim for 7-9 hours for better health.",
                "optimal": "💫 Great sleep duration! Keep maintaining this healthy sleep schedule.",
                "high": "🛏️ More than 9 hours of sleep. Consider consulting your healthcare provider."
            },
            
            "Hydration": {
                "yes": "💧 Well hydrated! Keep drinking water throughout the day.",
                "no": "🚰 Increase your water intake for better blood sugar control."
            }
        }

        self.create_user_interface()

    def animate_colors(self):
        """Animates the background color of buttons"""
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Button):
                widget.configure(bg=self.colors[self.current_color_index])
        
        self.current_color_index = (self.current_color_index + 1) % len(self.colors)
        self.root.after(1000, self.animate_colors)

    def create_user_interface(self):
        """Creates the user interface with improved animations and emojis"""
        # Animated welcome message
        welcome_text = "🌟 Welcome to Diabetes Management App 🌟"
        self.welcome_label = tk.Label(self.root, text="", font=self.header_font, bg="black", fg="white")
        self.welcome_label.pack(pady=20)
        self.animate_text(welcome_text, self.welcome_label)

        # Header Label
        tk.Label(self.root, text="👤 Enter User Details", font=self.header_font, bg="black", fg="white").pack(pady=20)

        # User Information Frame for better layout
        user_frame = tk.Frame(self.root, bg="black")
        user_frame.pack(pady=10)

        # User Information Fields
        self.create_label_and_entry(user_frame, "User ID", self.user_id, "This is your unique identifier.")
        self.create_label_and_entry(user_frame, "Name", self.user_name, "Your name helps to personalize the app experience.")
        self.create_label_and_entry(user_frame, "Age", self.age, "Age can influence your health risk factors.")
        self.create_label_and_entry(user_frame, "Weight (kg)", self.weight, "Your weight helps in determining your health status.")
        self.create_label_and_entry(user_frame, "Height (cm)", self.height, "Your height is used to calculate BMI.")
        
        # Submit Button
        submit_btn = tk.Button(self.root, text="Submit User Info", command=self.submit_user_info,
                               font=self.button_font, bg="#1abc9c", fg="black", activebackground="black", bd=3,
                               width=25, height=2)
        submit_btn.pack(pady=30)

        # Add animated emoji indicators
        self.status_label = tk.Label(self.root, text="💫", font=("Helvetica", 24), bg="black", fg="white")
        self.status_label.pack(side="bottom", pady=10)
        self.animate_status_emoji()

    def create_label_and_entry(self, parent_frame, label_text, text_var, tip_text):
        """ Utility function to create a label, entry widget, and tip for the user input form. """
        row = len(parent_frame.grid_slaves()) // 2  # Calculate current row to place fields
        tk.Label(parent_frame, text=f"{label_text} 📝", font=self.label_font, bg="black", fg="white").grid(row=row, column=0, sticky='e', padx=10, pady=10)
        entry = tk.Entry(parent_frame, textvariable=text_var, font=self.entry_font, relief="flat", width=22, bg="#2c3e50", fg="white")
        entry.grid(row=row, column=1, pady=10)
        entry.bind("<FocusIn>", lambda event: self.show_tip(tip_text))

    def show_tip(self, tip_text):
        """ Shows a tip to help users understand the importance of each input field. """
        tip_label = tk.Label(self.root, text=tip_text, font=self.tip_font, fg="yellow", bg="black")
        tip_label.pack(side="bottom", pady=5)
        self.root.after(5000, tip_label.destroy)  # Destroy the tip after 5 seconds

    def submit_user_info(self):
        """ Handles user information submission and proceeds to daily health data. """
        try:
            if not self.user_id.get() or not self.user_name.get():
                raise ValueError("User ID and Name are required.")
            if not self.age.get().isdigit() or int(self.age.get()) <= 0:
                raise ValueError("Age must be a positive integer.")
            if not self.weight.get().replace('.', '', 1).isdigit() or float(self.weight.get()) <= 0:
                raise ValueError("Weight must be a positive number.")
            if not self.height.get().replace('.', '', 1).isdigit() or float(self.height.get()) <= 0:
                raise ValueError("Height must be a positive number.")

            # Proceed to Daily Data Input Form
            self.show_daily_data_form()

        except ValueError as e:
            messagebox.showerror("Input Error", f"🚫 {e}")

    def show_daily_data_form(self):
        """Enhanced daily data form with animations"""
        # Clear with fade effect
        self.fade_out_widgets()
        
        # Daily Data Header
        tk.Label(self.root, text="📊 Enter Daily Health Data", font=self.header_font, bg="black", fg="white").pack(pady=20)

        # Daily Data Frame
        daily_frame = tk.Frame(self.root, bg="black")
        daily_frame.pack(pady=10)

        # Daily Data Fields
        self.create_label_and_entry(daily_frame, "Blood Glucose (mg/dL)", self.blood_glucose, "Blood glucose levels help monitor diabetes.")
        
        tk.Label(daily_frame, text="Diet 📝", font=self.label_font, bg="black", fg="white").grid(row=1, column=0, sticky='e', padx=10, pady=10)
        diet_combo = ttk.Combobox(daily_frame, textvariable=self.diet, font=self.entry_font, width=20)
        diet_combo['values'] = ["healthy", "unhealthy"]
        diet_combo.grid(row=1, column=1, pady=10)
        diet_combo.bind("<FocusIn>", lambda event: self.show_tip("Diet quality affects your blood sugar levels and overall health."))

        self.create_label_and_entry(daily_frame, "Physical Activity (minutes)", self.physical_activity, "Physical activity helps regulate your blood sugar and improve your health.")

        tk.Label(daily_frame, text="Medication Adherence 📝", font=self.label_font, bg="black", fg="white").grid(row=3, column=0, sticky='e', padx=10, pady=10)
        medication_combo = ttk.Combobox(daily_frame, textvariable=self.medication_adherence, font=self.entry_font, width=20)
        medication_combo['values'] = ["good", "poor"]
        medication_combo.grid(row=3, column=1, pady=10)
        medication_combo.bind("<FocusIn>", lambda event: self.show_tip("Taking your medications as prescribed is crucial for managing diabetes."))

        tk.Label(daily_frame, text="Stress Level 📝", font=self.label_font, bg="black", fg="white").grid(row=4, column=0, sticky='e', padx=10, pady=10)
        stress_combo = ttk.Combobox(daily_frame, textvariable=self.stress_level, font=self.entry_font, width=20)
        stress_combo['values'] = ["low", "medium", "high"]
        stress_combo.grid(row=4, column=1, pady=10)
        stress_combo.bind("<FocusIn>", lambda event: self.show_tip("Stress can impact blood sugar levels, so it's important to manage it."))

        self.create_label_and_entry(daily_frame, "Sleep Hours", self.sleep_hours, "Good sleep is crucial for blood sugar control and overall health.")

        tk.Label(daily_frame, text="Hydration Level 📝", font=self.label_font, bg="black", fg="white").grid(row=6, column=0, sticky='e', padx=10, pady=10)
        hydration_combo = ttk.Combobox(daily_frame, textvariable=self.hydration_level, font=self.entry_font, width=20)
        hydration_combo['values'] = ["yes", "no"]
        hydration_combo.grid(row=6, column=1, pady=10)
        hydration_combo.bind("<FocusIn>", lambda event: self.show_tip("Staying hydrated helps in regulating blood sugar levels."))

        # Submit Button for Daily Data
        submit_btn = tk.Button(self.root, text="Submit Daily Data", command=self.submit_daily_data,
                               font=self.button_font, bg="#e74c3c", fg="black", activebackground="#c0392b", bd=0,
                               width=20, height=2)
        submit_btn.pack(pady=30)

        # Add more emojis to labels
        daily_data_fields = [
            ("Blood Glucose", "🩺 Blood Glucose (mg/dL) 📊"),
            ("Diet", "🍎 Diet 🥗"),
            ("Physical Activity", "🏃‍♂️ Physical Activity (minutes) 🎯"),
            ("Medication Adherence", "💊 Medication Adherence 📅"),
            ("Stress Level", "🧘‍♂️ Stress Level 😌"),
            ("Sleep Hours", "😴 Sleep Hours 🌙"),
            ("Hydration Level", "💧 Hydration Level 🚰")
        ]

    def fade_out_widgets(self):
        """Creates a fade-out effect for widgets"""
        
        for widget in self.root.winfo_children():
            widget.pack_forget()
        self.root.update()
        time.sleep(0.3)

    def calculate_bmi(self):
        """Calculate BMI from weight and height"""
        try:
            weight = float(self.weight.get())
            height = float(self.height.get()) / 100  # Convert cm to meters
            bmi = weight / (height * height)
            self.bmi.set(f"{bmi:.1f}")
            return bmi
        except ValueError:
            return None

    def determine_activity_level(self, minutes):
        """Determine activity level based on minutes of physical activity"""
        minutes = float(minutes)
        if minutes < 30:
            return "low"
        elif minutes < 60:
            return "moderate"
        else:
            return "high"

    def submit_daily_data(self):
        try:
            # Calculate BMI
            bmi = self.calculate_bmi()
            if bmi is None:
                raise ValueError("Invalid weight or height values")

            # Validate Daily Data Inputs
            if not self.blood_glucose.get().isdigit() or int(self.blood_glucose.get()) <= 0:
                raise ValueError("Blood Glucose must be a positive integer.")
            if self.diet.get().lower() not in ["healthy", "unhealthy"]:
                raise ValueError("Diet must be 'healthy' or 'unhealthy'.")
            if not self.physical_activity.get().isdigit() or int(self.physical_activity.get()) < 0:
                raise ValueError("Physical activity must be a non-negative integer.")
            if self.medication_adherence.get().lower() not in ["good", "poor"]:
                raise ValueError("Medication adherence must be 'good' or 'poor'.")
            if self.stress_level.get().lower() not in ["low", "medium", "high"]:
                raise ValueError("Stress level must be 'low', 'medium', or 'high'.")
            if not self.sleep_hours.get().replace('.', '', 1).isdigit() or float(self.sleep_hours.get()) < 0:
                raise ValueError("Sleep hours must be a non-negative number.")
            if self.hydration_level.get().lower() not in ["yes", "no"]:
                raise ValueError("Hydration level must be 'yes' or 'no'.")

            # Map categorical variables to numeric values
            diet = 1 if self.diet.get().lower() == "healthy" else 0
            medication_adherence = 1 if self.medication_adherence.get().lower() == "good" else 0
            stress_level_map = {"low": 0, "medium": 1, "high": 2}
            stress_level = stress_level_map[self.stress_level.get().lower()]
            
            # Determine activity level from physical activity minutes
            activity_level = self.determine_activity_level(self.physical_activity.get())
            activity_level_map = {"low": 0, "moderate": 1, "high": 2}
            activity_level_numeric = activity_level_map[activity_level]
            hydration_level = 1 if self.hydration_level.get().lower() == "yes" else 0

            # Create input feature array
            features = np.array([[
                float(self.weight.get()),        # Weight
                float(self.height.get()),        # Height
                bmi,                             # BMI
                float(self.blood_glucose.get()), # Blood Glucose
                float(self.physical_activity.get()), # Physical Activity
                diet,                            # Diet
                medication_adherence,            # Medication Adherence
                stress_level,                    # Stress Level
                float(self.sleep_hours.get()),   # Sleep Hours
                hydration_level                  # Hydration Level
            ]])

            # Debug logging
            print("\nDebug Information:")
            print("Raw Features:", features)
            print("\nFeature Values:")
            print(f"Weight: {self.weight.get()} kg")
            print(f"Height: {self.height.get()} cm")
            print(f"BMI: {bmi}")
            print(f"Blood Glucose: {self.blood_glucose.get()} mg/dL")
            print(f"Physical Activity: {self.physical_activity.get()} minutes")
            print(f"Diet: {self.diet.get()} ({diet})")
            print(f"Medication Adherence: {self.medication_adherence.get()} ({medication_adherence})")
            print(f"Stress Level: {self.stress_level.get()} ({stress_level})")
            print(f"Sleep Hours: {self.sleep_hours.get()}")
            print(f"Activity Level: {activity_level} ({activity_level_numeric})")
            print(f"Hydration Level: {self.hydration_level.get()} ({hydration_level})")

            # Transform features using the saved scaler
            features_scaled = scaler.transform(features)
            print("Scaled Features:", features_scaled)
            
            # Predict risk score using the trained model
            risk_score = xgb_model.predict(features_scaled)[0]
            print("Predicted Risk Score:", risk_score)

            # Show feedback with BMI category and activity level
            bmi_category = (
                "Underweight" if bmi < 18.5 else
                "Normal weight" if bmi < 25 else
                "Overweight" if bmi < 30 else
                "Obese"
            )
            
            self.show_feedback(risk_score, bmi_category, self.physical_activity.get())

        except ValueError as e:
            messagebox.showerror("Input Error", f"🚫 {e}")
        except Exception as e:
            messagebox.showerror("Error", f"⚠️ An error occurred: {e}")

    def show_feedback(self, risk_score, bmi_category, physical_activity_minutes):
        """Enhanced interactive feedback with improved UI design"""
        activity_level = self.determine_activity_level(physical_activity_minutes)
        
        # Determine blood glucose status with adjusted ranges
        bg_level = float(self.blood_glucose.get())
        bg_status = "extreme" if bg_level > 300 else "low" if bg_level < 70 else "high" if bg_level > 180 else "normal"        
        # Determine sleep status
        sleep_hours = float(self.sleep_hours.get())
        sleep_status = "low" if sleep_hours < 7 else "high" if sleep_hours > 9 else "optimal"
        
        # Adjust risk score based on critical factors
        risk_modifiers = {
            'high_blood_glucose': 15 if bg_level > 180 else 0,
            'low_activity': 10 if float(physical_activity_minutes) < 30 else 0,
            'poor_medication': 10 if self.medication_adherence.get().lower() == "poor" else 0,
            'poor_sleep': 8 if sleep_hours < 6 else 0,
            'high_stress': 7 if self.stress_level.get().lower() == "high" else 0,
            'poor_hydration': 5 if self.hydration_level.get().lower() == "no" else 0
        }
        
        adjusted_risk_score = risk_score + sum(risk_modifiers.values())
        
        # Create feedback window with improved styling
        feedback_window = tk.Toplevel(self.root)
        feedback_window.title("Health Metrics Analysis 📊")
        feedback_window.geometry("900x1200")
        feedback_window.configure(bg="#1E1E1E")  # Darker, more professional background
        
        # Style configuration
        style = ttk.Style()
        style.configure("Custom.TFrame", background="#1E1E1E")
        style.configure("Custom.TButton", 
                       padding=10, 
                       font=("Helvetica", 12, "bold"),
                       background="#2E2E2E",
                       foreground="white")
        
        # Add header with gradient effect
        header_frame = tk.Frame(feedback_window, bg="#1E1E1E", height=100)
        header_frame.pack(fill='x', pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Gradient header label
        header_label = tk.Label(
            header_frame,
            text="Your Health Dashboard",
            font=("Helvetica", 28, "bold"),
            bg="#1E1E1E",
            fg="#00ff99",
            pady=20
        )
        header_label.pack(expand=True)

        # Determine risk level, color, and emoji
        risk_info = {
            "Low": {
                "color": "#00ff99",
                "emoji": "✅"
            },
            "Moderate": {
                "color": "#ffcc00",
                "emoji": "⚠️"
            },
            "High": {
                "color": "#ff3300",
                "emoji": "🚨"
            }
        }
        
        risk_level = "Low" if adjusted_risk_score < 30 else "Moderate" if adjusted_risk_score < 60 else "High"
        risk_color = risk_info[risk_level]["color"]
        risk_emoji = risk_info[risk_level]["emoji"]

        # Risk score display with improved visibility and emoji
        risk_frame = tk.Frame(feedback_window, bg="#2A2A2A", padx=30, pady=20)
        risk_frame.pack(fill='x', padx=40, pady=(0, 20))
        
        # Score display
        tk.Label(
            risk_frame,
            text=f"Risk Score: {adjusted_risk_score:.1f}",
            font=("Helvetica", 40, "bold"),
            bg="#2A2A2A",
            fg=risk_color
        ).pack(side='left', padx=20)
        
        # Risk level with emoji
        risk_level_frame = tk.Frame(risk_frame, bg="#2A2A2A")
        risk_level_frame.pack(side='right', padx=20)
        
        tk.Label(
            risk_level_frame,
            text=risk_emoji,
            font=("Helvetica", 24),
            bg="#2A2A2A",
            fg=risk_color
        ).pack()
        
        tk.Label(
            risk_level_frame,
            text=f"Risk Level: {risk_level}",
            font=("Helvetica", 24),
            bg="#2A2A2A",
            fg=risk_color
        ).pack()

        # Create scrollable content
        content_frame = tk.Frame(feedback_window, bg="#1E1E1E")
        content_frame.pack(fill='both', expand=True, padx=40)
        
        canvas = tk.Canvas(content_frame, bg="#1E1E1E", highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1E1E1E")

        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create metric cards with improved styling
        metrics = [
            {
                "title": "🏃 Physical Activity",
                "value": f"{activity_level.title()} ({physical_activity_minutes} min)",
                "advice": self.advice['Physical Activity'][activity_level]
            },
            {
                "title": "⚖️ BMI Status",
                "value": bmi_category,
                "advice": self.advice['BMI'][bmi_category]
            },
            {
                "title": "🩺 Blood Glucose",
                "value": f"{bg_level} mg/dL",
                "advice": self.advice['Blood Glucose'][bg_status]
            },
            {
                "title": "🍎 Diet",
                "value": self.diet.get(),
                "advice": self.advice['Diet'][self.diet.get().lower()]
            },
            {
                "title": "💊 Medication",
                "value": self.medication_adherence.get(),
                "advice": self.advice['Medication'][self.medication_adherence.get().lower()]
            },
            {
                "title": "😊 Stress Level",
                "value": self.stress_level.get(),
                "advice": self.advice['Stress'][self.stress_level.get().lower()]
            },
            {
                "title": "😴 Sleep",
                "value": f"{sleep_hours} hours",
                "advice": self.advice['Sleep'][sleep_status]
            },
            {
                "title": "💧 Hydration",
                "value": self.hydration_level.get(),
                "advice": self.advice['Hydration'][self.hydration_level.get().lower()]
            }
        ]

        for metric in metrics:
            card = tk.Frame(scrollable_frame, bg="#2A2A2A", padx=20, pady=15)
            card.pack(fill='x', pady=10)
            
            # Title and value in the same row
            header_frame = tk.Frame(card, bg="#2A2A2A")
            header_frame.pack(fill='x')
            
            tk.Label(
                header_frame,
                text=metric["title"],
                font=("Helvetica", 16, "bold"),
                bg="#2A2A2A",
                fg="white"
            ).pack(side='left')
            
            tk.Label(
                header_frame,
                text=metric["value"],
                font=("Helvetica", 16),
                bg="#2A2A2A",
                fg="#00ff99"
            ).pack(side='right')
            
            # Advice section
            tk.Label(
                card,
                text=metric["advice"],
                font=("Helvetica", 12),
                bg="#2A2A2A",
                fg="#cccccc",
                wraplength=700,
                justify='left'
            ).pack(pady=(10, 0), anchor='w')

        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Button frame with improved styling
        button_frame = tk.Frame(feedback_window, bg="#1E1E1E", pady=20)
        button_frame.pack(fill='x', padx=40, pady=20)

        # Enhanced buttons with black text
        share_btn = tk.Button(
            button_frame,
            text="Share with Doctor 👨‍⚕️",
            command=lambda: self.share_data_with_clinician(adjusted_risk_score),
            font=("Helvetica", 14, "bold"),
            bg="#2196F3",
            fg="black",  # Changed to black
            activebackground="#1976D2",
            activeforeground="black",  # Changed to black
            padx=30,
            pady=10,
            relief="flat",
            cursor="hand2"
        )
        share_btn.pack(side='left')

        close_btn = tk.Button(
            button_frame,
            text="Close Dashboard ✖️",
            command=feedback_window.destroy,
            font=("Helvetica", 14, "bold"),
            bg="#FF5252",
            fg="black",  # Changed to black
            activebackground="#D32F2F",
            activeforeground="black",  # Changed to black
            padx=30,
            pady=10,
            relief="flat",
            cursor="hand2"
        )
        close_btn.pack(side='right')

        # Show celebration for low risk
        if adjusted_risk_score < 30:
            self.show_celebration_animation()

    def share_data_with_clinician(self, risk_score):
        """ Simulate sharing patient's data with clinician's EHR."""
        # Convert data to JSON-serializable format
        fhir_data = {
            "user_id": self.user_id.get(),
            "name": self.user_name.get(),
            "age": int(self.age.get()),
            "records": [
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "blood_glucose": int(self.blood_glucose.get()),
                    "diet": self.diet.get(),
                    "physical_activity": int(self.physical_activity.get()),
                    "medication_adherence": self.medication_adherence.get(),
                    "stress_level": self.stress_level.get(),
                    "sleep_hours": float(self.sleep_hours.get()),
                    "hydration_level": self.hydration_level.get(),
                    "risk_score": float(risk_score)  # Convert to Python float to ensure JSON compatibility
                }
            ]
        }

        # Simulating HL7 FHIR data exchange by converting to JSON
        try:
            fhir_payload = json.dumps(fhir_data, indent=4)
            print(f"FHIR Data Exchange with EHR: {fhir_payload}")
            messagebox.showinfo("Data Sync", "✅ Data successfully shared with clinician's EHR!")
        except TypeError as e:
            messagebox.showerror("Serialization Error", f"🚫 Failed to serialize data: {e}")

    def animate_text(self, text, label, index=0):
        """Animates text appearing letter by letter"""
        if index < len(text):
            label.config(text=text[:index + 1])
            self.root.after(100, lambda: self.animate_text(text, label, index + 1))

    def animate_status_emoji(self):
        """Animates the status emoji"""
        emojis = ["💫", "✨", "🌟", "⭐", "🌠"]
        current = self.status_label.cget("text")
        next_emoji = emojis[(emojis.index(current) + 1) % len(emojis)]
        self.status_label.config(text=next_emoji)
        self.root.after(800, self.animate_status_emoji)

    def show_celebration_animation(self):
        """Shows a celebration animation"""
        celebration_emojis = ["🎉", "🎊", "✨", "💫", "🌟"]
        
        def animate_celebration(index=0):
            if index < len(celebration_emojis) * 3:  # Repeat 3 times
                emoji = celebration_emojis[index % len(celebration_emojis)]
                celebration_label = tk.Label(self.root, text=emoji, font=("Helvetica", 40), bg="black", fg="white")
                celebration_label.place(relx=0.5, rely=0.5, anchor="center")
                self.root.after(300, celebration_label.destroy)
                self.root.after(300, lambda: animate_celebration(index + 1))
        
        animate_celebration()

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesManagementApp(root)
    root.mainloop()

