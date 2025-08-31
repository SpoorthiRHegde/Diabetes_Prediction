from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import traceback
import io
import joblib
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

app = Flask(__name__, static_folder='templates', static_url_path='')
CORS(app)

MODEL_FILE = 'diabetes_model.pkl'
SCALER_FILE = 'diabetes_scaler.pkl'

# Mapping between form field names and model feature names
FEATURE_MAPPING = {
    'age': 'age',
    'sex': 'sex',
    'bmi': 'bmi',
    'highChol': 'highchol',
    'cholCheck': 'cholcheck',
    'smoker': 'smoker',
    'heartDisease': 'heartdiseaseorattack',
    'physActivity': 'physactivity',
    'fruits': 'fruits',
    'veggies': 'veggies',
    'alcohol': 'hvyalcoholconsump',
    'genHlth': 'genhlth',
    'mentHlth': 'menthlth',
    'physHlth': 'physhlth',
    'diffWalk': 'diffwalk',
    'stroke': 'stroke',
    'highBP': 'highbp'
}

def train_and_save_model():
    try:
        df = pd.read_csv('diabetes_data.csv')
        
        # Standardize column names (lowercase, no spaces)
        df.columns = df.columns.str.lower().str.replace(' ', '')
        if 'diabetes' not in df.columns:
            raise ValueError("Missing 'diabetes' column")

        X = df.drop('diabetes', axis=1)
        y = df['diabetes']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_res_scaled = scaler.fit_transform(X_res)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_res_scaled, y_res)
        y_pred = model.predict(X_test_scaled)

        print("✅ Model trained with SMOTE")
        print(f"📊 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("📉 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))

        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        
        return model, scaler

    except Exception as e:
        print(f"❌ Error in train_model: {e}")
        return None, None

def load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            print("✅ Model and scaler loaded from disk")
            return model, scaler
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None
    return None, None

# Try to load existing model, otherwise train new one
model, scaler = load_model()
if model is None or scaler is None:
    model, scaler = train_and_save_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON', 'status': 'error'}), 400

        data = request.get_json()
        
        # Get the model's expected feature names (from the scaler)
        expected_features = scaler.feature_names_in_
        
        # Create input DataFrame with correct feature names
        input_data = {}
        for form_field, model_feature in FEATURE_MAPPING.items():
            if form_field in data:
                input_data[model_feature] = data[form_field]
            else:
                return jsonify({'error': f'Missing required field: {form_field}', 'status': 'error'}), 400
        
        # Ensure the features are in the exact same order as during training
        input_df = pd.DataFrame([input_data])[expected_features]
        
        if model is None or scaler is None:
            return jsonify({'error': 'Model or scaler not loaded', 'status': 'error'}), 500

        input_scaled = scaler.transform(input_df)
        prediction = int(model.predict(input_scaled)[0])
        probability = round(float(model.predict_proba(input_scaled)[0][1]), 4)

        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'status': 'success'
        })

    except Exception as e:
        app.logger.error(f"Error in predict: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'status': 'error'}), 500


def format_field_name(field_name):
    field_map = {
        'age': 'Age',
        'sex': 'Sex',
        'bmi': 'BMI',
        'highChol': 'High Cholesterol',
        'cholCheck': 'Cholesterol Check (Past 5 Years)',
        'smoker': 'Smoker',
        'heartDisease': 'Heart Disease/Heart Attack',
        'physActivity': 'Physical Activity',
        'fruits': 'Daily Fruit Consumption',
        'veggies': 'Daily Vegetable Consumption',
        'alcohol': 'Heavy Alcohol Consumption',
        'genHlth': 'General Health Rating',
        'mentHlth': 'Poor Mental Health Days (Past 30)',
        'physHlth': 'Poor Physical Health Days (Past 30)',
        'diffWalk': 'Difficulty Walking/Climbing Stairs',
        'stroke': 'History of Stroke',
        'highBP': 'High Blood Pressure'
    }
    return field_map.get(field_name, field_name)


def format_field_value(field_name, value):
    try:
        val = float(value)
        if field_name == 'sex':
            return 'Male' if val == 1 else 'Female'
        elif field_name in [
            'highChol', 'cholCheck', 'smoker', 'heartDisease',
            'physActivity', 'fruits', 'veggies', 'alcohol',
            'diffWalk', 'stroke', 'highBP'
        ]:
            return 'Yes' if val == 1 else 'No'
        elif field_name == 'genHlth':
            health_map = {1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'}
            return health_map.get(int(val), str(val))
        elif field_name in ['age', 'mentHlth', 'physHlth']:
            return f"{int(val)} days" if 'Hlth' in field_name else str(int(val))
        elif field_name == 'bmi':
            return f"{val:.1f}"
        else:
            return str(val)
    except:
        return str(value)


@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON', 'status': 'error'}), 400

        data = request.get_json()
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1 * inch)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Heading1'], fontSize=24,
            textColor=colors.HexColor('#4a6fa5'), spaceAfter=30, alignment=TA_CENTER
        )
        heading_style = ParagraphStyle(
            'CustomHeading', parent=styles['Heading2'], fontSize=16,
            textColor=colors.HexColor('#166088'), spaceAfter=12, spaceBefore=20
        )

        story = [Paragraph("Health Risk Assessment Report", title_style), Spacer(1, 12)]
        story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
        story.append(Spacer(1, 20))

        story.append(Paragraph("Risk Assessment Results", heading_style))
        prediction = data.get('prediction', 0)
        probability = data.get('probability', 0)
        risk_status = 'High Risk' if prediction else 'Low Risk'
        risk_percentage = f"{probability * 100:.1f}%"
        risk_color = colors.red if prediction else colors.green
        risk_text = f"<b><font color='{risk_color}'>Risk Level: {risk_status} ({risk_percentage})</font></b>"
        story.append(Paragraph(risk_text, styles['Normal']))
        story.append(Spacer(1, 20))

        story.append(Paragraph("Personal Health Information", heading_style))
        table_data = [['Health Factor', 'Your Response']]
        excluded_fields = ['timestamp', 'prediction', 'probability']
        for key, value in data.items():
            if key not in excluded_fields:
                table_data.append([format_field_name(key), format_field_value(key, value)])

        table = Table(table_data, colWidths=[3 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a6fa5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        story.append(table)
        story.append(Spacer(1, 30))

        story.append(Paragraph("Health Recommendations", heading_style))
        recommendations = [
            "Consult with your healthcare provider about your risk factors",
            "Consider regular health screenings and check-ups",
            "Adopt a healthier diet with more fruits and vegetables",
            "Increase physical activity to at least 150 minutes per week",
            "If you smoke, consider quitting",
            "Monitor your blood pressure and cholesterol regularly",
            "Limit alcohol consumption",
            "Manage stress through relaxation techniques"
        ] if prediction == 1 else [
            "Continue with your healthy habits",
            "Maintain regular physical activity",
            "Keep up with balanced nutrition",
            "Schedule regular health check-ups",
            "Monitor any changes in your health status",
            "Stay aware of your family health history"
        ]

        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
            story.append(Spacer(1, 6))

        story.append(Spacer(1, 30))
        story.append(Paragraph("Important Disclaimer", heading_style))
        disclaimer_text = """
        This health risk assessment is for informational purposes only and should not be considered 
        as medical advice, diagnosis, or treatment. The results are based on statistical models and 
        may not reflect your actual health status. Always consult with qualified healthcare 
        professionals for proper medical evaluation and personalized health advice.
        """
        story.append(Paragraph(disclaimer_text, styles['Normal']))
        doc.build(story)

        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name='health_risk_assessment_report.pdf',
            mimetype='application/pdf'
        )

    except Exception as e:
        app.logger.error(f"Error in generate_report: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'status': 'error'}), 500


if __name__ == '__main__':
    app.run(debug=True)
