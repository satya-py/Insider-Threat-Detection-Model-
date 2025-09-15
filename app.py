from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from utils.insider_threat_model_loader import InsiderThreatModelLoader
import os

# Create Flask application instance
app = Flask(__name__)

# Set secret key for session management and flash messages
app.secret_key = 'your-secret-key-change-this-in-production'

# Initialize the model loader
model_loader = InsiderThreatModelLoader()

# Load all model components when the application starts
print("Loading model components...")
if not model_loader.load_all_components():
    print("Warning: Failed to load some model components!")


@app.route('/')
def home():
    """
    Home page route - displays the main landing page.
    This is the first page users see when they visit the application.
    """
    return render_template('home.html')


@app.route('/predict')
def predict_page():
    """
    Prediction page route - displays the form for inputting features.
    This page contains the form where users enter employee data for analysis.
    """
    # Get feature names to display in the form
    feature_names = model_loader.get_feature_names()

    # Check if model is ready for predictions
    model_ready = model_loader.is_ready()

    return render_template('index.html',
                           feature_names=feature_names,
                           model_ready=model_ready)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests when form is submitted.
    This processes the form data and returns the threat assessment.
    """
    try:
        # Check if model is loaded and ready
        if not model_loader.is_ready():
            flash('Model is not ready. Please check model files.', 'error')
            return redirect(url_for('predict_page'))

        # Get form data - extract all the feature values from the submitted form
        form_data = request.form.to_dict()

        # Convert form data to proper format for model input
        input_data = {}

        # Process each feature from the form
        for feature_name in model_loader.get_feature_names():
            if feature_name in form_data:
                try:
                    # Convert string input to float (all features are numeric)
                    input_data[feature_name] = float(form_data[feature_name])
                except ValueError:
                    # If conversion fails, set to 0 as default
                    input_data[feature_name] = 0.0
            else:
                # If feature not provided, set to 0
                input_data[feature_name] = 0.0

        # Make prediction using the loaded model
        prediction, probability = model_loader.predict(input_data)

        # Check if prediction was successful
        if prediction is None:
            flash('Error making prediction. Please check your input data.', 'error')
            return redirect(url_for('predict_page'))

        # Interpret the results
        if prediction == 1:
            threat_status = "HIGH RISK"
            threat_class = "danger"
            message = "This employee profile shows high risk indicators for insider threat."
        else:
            threat_status = "LOW RISK"
            threat_class = "success"
            message = "This employee profile shows low risk for insider threat."

        # Format probability as percentage
        confidence_percentage = round(probability * 100, 2)

        # Prepare result data to send to template
        result = {
            'prediction': prediction,
            'probability': probability,
            'confidence_percentage': confidence_percentage,
            'threat_status': threat_status,
            'threat_class': threat_class,
            'message': message,
            'input_data': input_data
        }

        # Render the results page with prediction results
        return render_template('index.html',
                               result=result,
                               feature_names=model_loader.get_feature_names(),
                               model_ready=True)

    except Exception as e:
        # Handle any unexpected errors
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('predict_page'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions - allows programmatic access to the model.
    This enables other applications to send JSON data and get predictions back.
    """
    try:
        # Check if model is ready
        if not model_loader.is_ready():
            return jsonify({
                'error': 'Model is not ready',
                'success': False
            }), 500

        # Get JSON data from request
        json_data = request.get_json()

        if not json_data:
            return jsonify({
                'error': 'No JSON data provided',
                'success': False
            }), 400

        # Make prediction
        prediction, probability = model_loader.predict(json_data)

        if prediction is None:
            return jsonify({
                'error': 'Error making prediction',
                'success': False
            }), 500

        # Return JSON response
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence_percentage': round(probability * 100, 2),
            'threat_status': 'HIGH RISK' if prediction == 1 else 'LOW RISK',
            'success': True
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/about')
def about():
    """
    About page route - displays information about the model and features.
    This helps users understand what the model does and how to use it.
    """
    # Updated feature descriptions to match your specific features
    feature_descriptions = {
        'logon_cnt': 'Total number of login attempts by the employee',
        'http_cnt': 'Total number of HTTP requests made by the employee',
        'email_cnt': 'Total number of emails sent by the employee',
        'usb_cnt': 'Total number of USB device connections',
        'logon_after_hours_pct': 'Percentage of logins that occurred after business hours (0-100)',
        'logon_weekend_cnt': 'Number of logins that occurred on weekends',
        'logon_hour_entropy': 'Randomness in login hours (higher = more unpredictable timing)',
        'logon_dow_entropy': 'Randomness in login days of week (higher = more irregular pattern)',
        'usb_act_Connect': 'Number of USB device connect actions recorded',
        'usb_act_Disconnect': 'Number of USB device disconnect actions recorded',
        'usb_after_hours_pct': 'Percentage of USB usage that occurred after business hours (0-100)',
        'usb_weekend_cnt': 'Number of USB connections that occurred on weekends',
        'email_to_http_ratio': 'Ratio of email activity to HTTP activity (email_cnt/http_cnt)',
        'usb_to_logon_ratio': 'Ratio of USB connections to login attempts (usb_cnt/logon_cnt)',
        'O': 'Openness to Experience - personality trait score (typically 1-5)',
        'C': 'Conscientiousness - personality trait score (typically 1-5)',
        'E': 'Extraversion - personality trait score (typically 1-5)',
        'A': 'Agreeableness - personality trait score (typically 1-5)',
        'N': 'Neuroticism - personality trait score (typically 1-5)',
        'psych_risk': 'Psychological risk assessment score (higher = more risk)'
    }

    return render_template('about.html',
                           feature_descriptions=feature_descriptions,
                           total_features=len(model_loader.get_feature_names()))


if __name__ == '__main__':
    """
    Run the Flask application in debug mode.
    Debug mode provides helpful error messages and auto-reloads when code changes.
    """
    # Check if all model components loaded successfully before starting
    if model_loader.is_ready():
        print(" All model components loaded successfully!")
        print(" Starting Flask application...")
    else:
        print("  Warning: Some model components failed to load!")
        print(" Starting Flask application anyway...")

    # Start the Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)