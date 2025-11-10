# Diabetes Management Application 🏥

**Course:** Health Informatics (SI 542) | **University of Michigan**

An intelligent healthcare application that leverages machine learning to help users monitor and manage their diabetes risk factors. Built with Python/Tkinter, it provides real-time risk assessment, personalized health recommendations, and FHIR-compatible data exchange for seamless EHR integration.

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

```
├── app.py                              # Main application interface
├── datapreprocessing&modeltrainning.py # ML pipeline and model training
├── diabetes_risk_model.pkl             # Trained XGBoost model
├── scaler.pkl                          # Feature scaler for preprocessing
├── diabetes_data.csv                   # Training dataset (from Kaggle)
├── plots/                              # Model evaluation visualizations
│   ├── feature_importance.png
│   └── actual_vs_predicted.png
└── README.md
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.7+
- Required packages (see below)

### Installation Steps

1. **Clone or download the repository**

2. **Install dependencies:**
```bash
pip install numpy pandas scikit-learn xgboost matplotlib tk joblib
```

3. **Download the training dataset:**
   - Download `diabetes_data.csv` from [Kaggle Dataset](https://bit.ly/3LoRKdt)
   - Place the CSV file in the project root directory

4. **Train the model** (if needed):
```bash
python datapreprocessing&modeltrainning.py
```
   Note: Pre-trained model files (`diabetes_risk_model.pkl` and `scaler.pkl`) are included in the repository.

5. **Launch the application:**
```bash
python app.py
```

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

### 1. User Profile Setup
- Enter User ID and Name
- Input Age, Weight (kg), and Height (cm)
- Click "Submit User Info"

### 2. Daily Health Data Entry
- Enter current Blood Glucose level
- Select Diet quality (healthy/unhealthy)
- Input Physical Activity duration (minutes)
- Select Medication Adherence status
- Choose Stress Level
- Enter Sleep Hours
- Select Hydration Status
- Click "Submit Daily Data"

### 3. View Risk Assessment
- Review Risk Score and Risk Level
- Explore detailed health metric cards
- Read personalized recommendations
- Share data with clinician (FHIR export)

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
