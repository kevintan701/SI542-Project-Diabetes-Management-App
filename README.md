# Diabetes Management Application 🏥

**Course:** Health Informatics (SI 542) | **University of Michigan**

An intelligent healthcare application that leverages machine learning to help users monitor and manage their diabetes risk factors. Built with Python/Tkinter, it provides real-time risk assessment, personalized health recommendations, and FHIR-compatible data exchange for seamless EHR integration.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
  - [Key Achievements](#key-achievements)
- [Features](#-features)
  - [Core Functionality](#core-functionality)
- [Technical Stack](#️-technical-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Installation Guide](#step-by-step-installation-guide)
  - [Troubleshooting](#troubleshooting)
- [Model Details](#-model-details)
  - [Machine Learning Pipeline](#machine-learning-pipeline)
- [Usage Guide](#-usage-guide)
  - [Quick Start](#quick-start)
  - [Detailed Usage Instructions](#detailed-usage-instructions)
- [Healthcare Integration](#-healthcare-integration)
  - [FHIR Data Format](#fhir-data-format)
  - [Risk Assessment Logic](#risk-assessment-logic)
- [Key Technical Highlights](#-key-technical-highlights)
- [Learning Outcomes](#-learning-outcomes)
- [Team](#-team)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Additional Resources](#-additional-resources)
  - [Research References](#research-references)

---

## 🎯 Project Overview

This application demonstrates the integration of machine learning, healthcare informatics, and user-centered design to create a practical diabetes management tool. The system uses an XGBoost regression model trained on comprehensive health metrics to predict diabetes risk scores and provide actionable insights to users and healthcare providers.

### Key Achievements

- **Machine Learning Integration**: Implemented XGBoost regression model with cross-validation achieving 97% accuracy (cross-validation score: 0.97 ± 0.02)
- **Healthcare Standards Compliance**: FHIR-compatible data formatting for EHR integration
- **User-Centered Design**: Intuitive GUI with real-time validation, interactive feedback, and personalized recommendations
- **End-to-End Pipeline**: Complete workflow from data preprocessing to model deployment
- **Clinical Relevance**: Risk assessment based on evidence-based medical guidelines

---

## 🚀 Features

### Core Functionality
- **👤 Smart Profile Management**: User information collection with real-time input validation
- **📊 Daily Health Tracking**: Comprehensive monitoring of 8+ health metrics
  - Blood Glucose (70-300 mg/dL)
  - Physical Activity (minutes/day)
  - Diet Quality (healthy/unhealthy)
  - Medication Adherence (good/poor)
  - Stress Level (low/medium/high)
  - Sleep Duration (hours)
  - Hydration Status (yes/no)
  - BMI calculation and categorization

- **🤖 ML-Powered Risk Assessment**: Real-time diabetes risk prediction using trained XGBoost model
  - Risk categories: Low (<30), Moderate (30-60), High (>60)
  - Dynamic risk adjustment based on critical health factors
  - Visual feedback with color-coded indicators

- **💡 Personalized Health Recommendations**: Context-aware advice for each health metric
- **🏥 EHR Integration**: FHIR-compatible data export for clinician sharing
- **🎨 Modern UI/UX**: Dark theme interface with animations and interactive elements

---

## 🛠️ Technical Stack

### Frontend
- **Python Tkinter**: Desktop GUI framework
- **Custom Widgets**: Interactive metric cards, scrollable dashboards, animated feedback

### Backend & Machine Learning
- **XGBoost**: Gradient boosting regression model
- **scikit-learn**: Data preprocessing, feature scaling, cross-validation
- **NumPy & Pandas**: Data manipulation and numerical computations
- **Joblib**: Model persistence and loading

### Data & Standards
- **HL7 FHIR**: Healthcare data exchange format (JSON)
- **CSV**: Data storage and processing
- **Matplotlib**: Feature importance visualization

---

## 📁 Project Structure

Understanding the project structure helps you navigate and work with the codebase effectively.

```
SI542-Project-Diabetes-Management-App/
├── app.py                              # Main application interface (GUI)
├── datapreprocessing&modeltrainning.py # ML pipeline and model training script
├── diabetes_risk_model.pkl             # Pre-trained XGBoost model (258 KB)
├── scaler.pkl                          # Feature scaler for data preprocessing
├── diabetes_data.csv                   # Training dataset (download from Kaggle)
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Git ignore rules
├── plots/                              # Model evaluation visualizations
│   ├── feature_importance.png         # Feature importance analysis
│   └── actual_vs_predicted.png        # Model performance visualization
└── README.md                           # This file - project documentation
```

### File Descriptions

- **`app.py`**: Main application file containing the Tkinter GUI and user interface logic. This is the entry point for running the application.

- **`datapreprocessing&modeltrainning.py`**: Machine learning pipeline script that handles data preprocessing, model training, and evaluation. Run this to retrain the model with new data.

- **`diabetes_risk_model.pkl`**: Serialized XGBoost model file. Contains the trained machine learning model used for risk prediction.

- **`scaler.pkl`**: StandardScaler object used to normalize input features before prediction. Must be used with the corresponding model.

- **`diabetes_data.csv`**: Training dataset (not included in repository). Download from [Kaggle](https://bit.ly/3LoRKdt) and place in the project root.

- **`requirements.txt`**: Lists all Python package dependencies with version specifications. Use `pip install -r requirements.txt` to install all dependencies.

- **`plots/`**: Directory containing visualization outputs from model training, including feature importance charts and prediction accuracy plots.

---

## 🔧 Installation & Setup

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.7 or higher** - Check your version with `python3 --version`
- **pip** - Python package installer (usually comes with Python)
- **Git** (optional) - For cloning the repository

### Step-by-Step Installation Guide

Follow these instructions carefully to set up the application on your local machine.

#### Step 1: Clone or Download the Repository

**Option A: Using Git (Recommended)**
```bash
git clone https://github.com/kevintan701/SI542-Project-Diabetes-Management-App.git
cd SI542-Project-Diabetes-Management-App
```

**Option B: Manual Download**
1. Click the green "Code" button on the GitHub repository page
2. Select "Download ZIP"
3. Extract the ZIP file to your desired location
4. Open a terminal and navigate to the extracted folder

#### Step 2: Install Python Dependencies

Install all required packages using pip:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib tk joblib
```

**Note for macOS users:** If you encounter an "externally-managed-environment" error, use:
```bash
pip install --break-system-packages numpy pandas scikit-learn xgboost matplotlib tk joblib
```

**Alternative: Using requirements.txt**
```bash
pip install -r requirements.txt
```

#### Step 3: Download the Training Dataset

The application requires a training dataset to function properly:

1. **Download the dataset:**
   - Visit the [Kaggle Dataset](https://bit.ly/3LoRKdt)
   - Click "Download" to get the `diabetes_data.csv` file
   - **Note:** You may need to create a free Kaggle account if prompted

2. **Place the file:**
   - Move `diabetes_data.csv` to the project root directory (same folder as `app.py`)
   - Verify the file is in the correct location:
     ```bash
     ls diabetes_data.csv
     ```

#### Step 4: Verify Model Files (Optional)

The repository includes pre-trained model files, so you typically don't need to retrain:

- ✅ `diabetes_risk_model.pkl` - Pre-trained XGBoost model
- ✅ `scaler.pkl` - Feature scaler for data preprocessing

**If you want to retrain the model:**
```bash
python datapreprocessing&modeltrainning.py
```
This will regenerate the model files based on your dataset.

#### Step 5: Launch the Application

Run the main application:

```bash
python app.py
```

Or on some systems:
```bash
python3 app.py
```

The application window should open automatically. If you encounter any errors, see the Troubleshooting section below.

### Troubleshooting

**Problem: "ModuleNotFoundError: No module named 'numpy'"**
- **Solution:** Install missing packages using `pip install numpy pandas scikit-learn xgboost matplotlib tk joblib`

**Problem: "FileNotFoundError: diabetes_risk_model.pkl"**
- **Solution:** Ensure you're running the script from the project root directory, or retrain the model using Step 4

**Problem: "Tkinter not found" (Linux)**
- **Solution:** Install tkinter: `sudo apt-get install python3-tk` (Ubuntu/Debian) or `sudo yum install python3-tkinter` (CentOS/RHEL)

**Problem: Application window doesn't appear**
- **Solution:** Check terminal for error messages. Ensure all dependencies are installed correctly.

**Problem: "Permission denied" errors**
- **Solution:** On macOS/Linux, you may need to use `python3` instead of `python`, or adjust file permissions

---

## 📊 Model Details

### Machine Learning Pipeline

**Model Type**: XGBoost Regressor  
**Training Approach**: 80-20 train-test split with cross-validation  
**Model Performance**: Cross-validation score of 0.97 ± 0.02 (97% accuracy)  
**Dataset Source**: [Kaggle - Diabetes Prediction Datasets](https://bit.ly/3LoRKdt)  
**Feature Engineering**:
- StandardScaler normalization
- Categorical variable encoding
- BMI calculation from weight/height
- Activity level categorization

**Input Features** (10 features):
1. Weight (kg)
2. Height (cm)
3. BMI (calculated)
4. Blood Glucose (mg/dL)
5. Physical Activity (minutes)
6. Diet (binary: healthy/unhealthy)
7. Medication Adherence (binary: good/poor)
8. Stress Level (categorical: low/medium/high)
9. Sleep Hours
10. Hydration Level (binary: yes/no)

**Evaluation Metrics**:
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Cross-validation scores

**Model Artifacts**:
- `diabetes_risk_model.pkl`: Trained model
- `scaler.pkl`: Feature scaler
- `feature_importance.png`: Feature importance visualization

---

## 💻 Usage Guide

This section provides comprehensive instructions on how to use the Diabetes Management Application effectively.

### Quick Start

1. **Launch the application** (see [Installation & Setup](#-installation--setup))
2. **Enter your profile information**
3. **Input daily health metrics**
4. **Review your risk assessment and recommendations**

### Detailed Usage Instructions

#### Step 1: User Profile Setup

When you first launch the application, you'll see the user profile form:

1. **Enter User ID:**
   - Create a unique identifier (e.g., "USER001" or your initials)
   - This helps track your health data over time

2. **Enter Your Name:**
   - Your name for personalization (e.g., "John Doe")

3. **Enter Age:**
   - Input your age in years (must be a positive number)
   - Example: `35`

4. **Enter Weight:**
   - Input your weight in kilograms (kg)
   - Example: `75.5` (for 75.5 kg)
   - **Conversion:** 1 kg = 2.2 lbs

5. **Enter Height:**
   - Input your height in centimeters (cm)
   - Example: `175` (for 175 cm or 5'9")
   - **Conversion:** 1 cm = 0.3937 inches

6. **Click "Submit User Info"**
   - The application will validate your inputs
   - If successful, you'll proceed to the daily health data form

**Tips:**
- Ensure all fields are filled correctly
- Use decimal numbers for weight (e.g., 75.5, not 75)
- The application will automatically calculate your BMI

#### Step 2: Daily Health Data Entry

After submitting your profile, you'll see the daily health tracking form:

1. **Blood Glucose Level:**
   - Enter your current blood glucose reading in mg/dL
   - Normal range: 70-100 mg/dL (fasting)
   - Acceptable range: 70-180 mg/dL (after meals)
   - Example: `95` or `140`

2. **Diet Quality:**
   - Select from dropdown: `healthy` or `unhealthy`
   - **Healthy:** Balanced meals with vegetables, whole grains, lean proteins
   - **Unhealthy:** Processed foods, high sugar, excessive carbs

3. **Physical Activity:**
   - Enter minutes of exercise/physical activity per day
   - Include walking, running, gym workouts, etc.
   - Example: `30` (for 30 minutes)
   - **Recommendation:** Aim for at least 30 minutes daily

4. **Medication Adherence:**
   - Select from dropdown: `good` or `poor`
   - **Good:** Taking medications as prescribed
   - **Poor:** Missing doses or not following prescription

5. **Stress Level:**
   - Select from dropdown: `low`, `medium`, or `high`
   - Consider your overall stress from work, life, health concerns

6. **Sleep Hours:**
   - Enter average hours of sleep per night
   - Example: `7.5` (for 7.5 hours)
   - **Recommendation:** 7-9 hours for adults

7. **Hydration Status:**
   - Select from dropdown: `yes` or `no`
   - **Yes:** Drinking adequate water throughout the day
   - **No:** Not drinking enough water

8. **Click "Submit Daily Data"**
   - The application will process your inputs
   - Your risk assessment will be calculated and displayed

#### Step 3: View Risk Assessment

After submitting your daily data, a comprehensive dashboard will appear:

1. **Risk Score Display:**
   - View your calculated risk score (0-100 scale)
   - See color-coded risk level:
     - 🟢 **Low Risk** (<30): Green indicator
     - 🟡 **Moderate Risk** (30-60): Yellow indicator
     - 🔴 **High Risk** (>60): Red indicator

2. **Health Metric Cards:**
   - Scroll through detailed cards for each health metric
   - Each card shows:
     - Current value/status
     - Personalized recommendations
     - Actionable advice

3. **Personalized Recommendations:**
   - Read context-specific advice for each metric
   - Follow suggestions to improve your health outcomes

4. **Share with Clinician:**
   - Click "Share with Doctor 👨‍⚕️" button
   - Your data will be exported in FHIR-compatible JSON format
   - A `fhir_data.json` file will be created in the project directory
   - You can share this file with your healthcare provider

**Understanding Your Results:**
- **Low Risk:** Continue maintaining healthy habits
- **Moderate Risk:** Consider lifestyle adjustments and regular monitoring
- **High Risk:** Consult with healthcare provider and follow recommendations

**Best Practices:**
- Enter data daily for accurate tracking
- Be honest about your health metrics
- Review recommendations regularly
- Share data with your healthcare provider during checkups

---

## 🏥 Healthcare Integration

### FHIR Data Format
The application exports patient data in HL7 FHIR-compatible JSON format, enabling seamless integration with Electronic Health Records (EHR) systems. Data includes:
- Patient demographics
- Health metrics with timestamps
- Calculated risk scores
- Structured for clinical decision support

### Risk Assessment Logic
Risk scores are calculated using:
- **Base Prediction**: XGBoost model output
- **Risk Modifiers**: Additional adjustments based on critical factors:
  - High blood glucose (>180 mg/dL): +15 points
  - Low physical activity (<30 min): +10 points
  - Poor medication adherence: +10 points
  - Poor sleep (<6 hours): +8 points
  - High stress: +7 points
  - Poor hydration: +5 points

---

## 📈 Key Technical Highlights

1. **Robust Data Pipeline**: Automated data preprocessing, validation, and feature engineering
2. **Model Performance**: Cross-validated XGBoost model achieving 97% accuracy with comprehensive evaluation metrics
3. **Real-time Processing**: Instant risk calculation with feature scaling and normalization
4. **Error Handling**: Comprehensive input validation and user-friendly error messages
5. **Modular Architecture**: Separated concerns (data preprocessing, model training, application interface)
6. **Healthcare Standards**: FHIR-compliant data formatting for interoperability
7. **Kaggle Dataset Integration**: Trained on real-world diabetes risk factors dataset

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **Machine Learning**: Model selection, training, evaluation, and deployment
- **Healthcare Informatics**: FHIR standards, EHR integration, clinical decision support
- **Software Engineering**: Object-oriented design, GUI development, data pipelines
- **Data Science**: Feature engineering, preprocessing, visualization
- **User Experience**: Interactive design, real-time feedback, personalized recommendations

---

## 👥 Team

**Course**: SI 542 - Health Informatics  
**Institution**: University of Michigan

**Team Members**:
- Kevin Tan
- Haichao Min
- Hanfu Hou
- Shreyas Karnad
- You Wu
- Donald Su

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- SI 542 Course Staff at University of Michigan
- Healthcare professionals who provided domain expertise
- Open-source community for various dependencies and tools

---

## 📚 Additional Resources

- **Demo Video**: `Diabete Management App - Demo Video.mp4`
- **Model Visualizations**: See `plots/` directory
- **Training Dataset**: [Kaggle - Diabetes Prediction Datasets](https://bit.ly/3LoRKdt)

### Research References

1. **Cardiovascular Health Management in Diabetic Patients with Machine-Learning-Driven Predictions and Interventions**  
   R. Jose, F. Syed, A. Thomas, and M. Toma, *Applied Sciences*, vol. 14, no. 5, pp. 2132-2145, 2024.  
   [Link](https://www.mdpi.com/2076-3417/14/5/2132)

2. **Type 2 Diabetes: Causes and Risk Factors**  
   S. Seed, *WebMD*, Aug. 16, 2024.  
   [Link](https://www.webmd.com/diabetes/diabetes-causes)

3. **Risk Factors for Type 2 Diabetes**  
   National Institute of Diabetes and Digestive and Kidney Diseases.  
   [Link](https://www.niddk.nih.gov/health-information/diabetes/overview/risk-factors-type-2-diabetes)

4. **Physical Activity/Exercise and Diabetes: A Position Statement of the American Diabetes Association**  
   R. J. Sigal et al., *Diabetes Care*, vol. 39, no. 11, pp. 2065-2079, Nov. 2016.  
   [Link](https://diabetesjournals.org/care/article/39/11/2065/37249/Physical-Activity-Exercise-and-Diabetes-A-Position)

5. **Environmental/Lifestyle Factors in the Pathogenesis and Prevention of Type 2 Diabetes**  
   H. Kolb and S. Martin, *BMC Medicine*, vol. 15, no. 1, p. 131, Jul. 2017.  
   [Link](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-017-0901-x)

6. **Determinants of Adherence to Diabetes Medications: Findings From a Large Pharmacy Claims Database**  
   M. S. Kirkman et al., *Diabetes Care*, vol. 38, no. 4, pp. 604-609, 2015.  
   [Link](https://diabetesjournals.org/care/article/38/4/604/37531/Determinants-of-Adherence-to-Diabetes-Medications)

7. **The Metabolic Consequences of Sleep Deprivation**  
   K. L. Knutson et al., *Sleep Medicine Reviews*, vol. 11, no. 3, pp. 163-178, Jun. 2007.  
   [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC1991337/)

8. **Sleep Duration and Diabetes Risk in American Indian and Alaska Native Participants of a Lifestyle Intervention Project**  
   D. S. Nuyujukian et al., *Sleep*, vol. 39, no. 11, pp. 1919-1926, Nov. 2016.  
   [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC5070746/)

9. **Diabetes and Hydration: Are You Drinking Enough Water?**  
   E. Davis, *diaTribe*, Jan. 29, 2024.  
   [Link](https://diatribe.org/diet-and-nutrition/diabetes-and-hydration-are-you-drinking-enough-water)

---

*This project was developed as part of the Health Informatics course (SI 542) at the University of Michigan, demonstrating the application of machine learning and healthcare informatics principles to real-world diabetes management challenges.*
