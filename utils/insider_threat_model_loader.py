import joblib
import json
import numpy as np
import os


class InsiderThreatModelLoader:
    """
    A class to load and manage the insider threat detection model and its components.
    This handles loading the trained model, scaler, vectorizer, and feature names.
    """

    def __init__(self, models_path='models/'):
        """
        Initialize the model loader with the path to models directory.

        Args:
            models_path (str): Path to the directory containing model files
        """
        self.models_path = models_path
        self.model = None
        self.scaler = None
        self.tfidf_vectorizer = None
        self.numeric_features = None

    def load_model(self):
        """
        Load the trained insider threat detection model from joblib file.
        This is the main ML model that makes predictions.
        """
        try:
            # Updated path to match your exact filename
            model_path = os.path.join(self.models_path, 'insiderthreat model(joblib file).joblib')
            self.model = joblib.load(model_path)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def load_scaler(self):
        """
        Load the feature scaler used during training.
        This ensures input data is normalized the same way as training data.
        """
        try:
            # Try multiple possible scaler file names
            possible_scaler_files = [
                'scaler(joblib).joblib',
                'scaler.joblib',
                'standard_scaler.joblib',
                'feature_scaler.joblib'
            ]

            scaler_loaded = False
            for filename in possible_scaler_files:
                scaler_path = os.path.join(self.models_path, filename)
                if os.path.exists(scaler_path):
                    try:
                        self.scaler = joblib.load(scaler_path)
                        print(f"Scaler loaded successfully from: {filename}")
                        scaler_loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load {filename}: {e}")
                        continue

            if not scaler_loaded:
                print("No scaler file found. Using raw features without scaling.")
                self.scaler = None

            return True  # Don't fail the app if scaler is missing

        except Exception as e:
            print(f"Error in scaler loading process: {e}")
            self.scaler = None
            return True  # Don't fail the app

    def load_tfidf_vectorizer(self):
        """
        Load the TF-IDF vectorizer for text features (if any).
        This converts text data into numerical features.
        """
        try:
            tfidf_path = os.path.join(self.models_path, 'tfidf_vectorizer(joblib).joblib')
            if os.path.exists(tfidf_path):
                try:
                    self.tfidf_vectorizer = joblib.load(tfidf_path)
                    print("TF-IDF Vectorizer loaded successfully!")
                    return True
                except Exception as e:
                    print(f"TF-IDF vectorizer exists but failed to load: {e}")
                    print("This might be due to custom tokenizer or version incompatibility.")
                    self.tfidf_vectorizer = None
                    return True  # Don't fail the app
            else:
                print("No TF-IDF vectorizer file found. Skipping text processing.")
                self.tfidf_vectorizer = None
                return True

        except Exception as e:
            print(f"Error in TF-IDF loading process: {e}")
            self.tfidf_vectorizer = None
            return True  # Don't fail the app

    def load_feature_names(self):
        """
        Load the list of numeric feature names from JSON file.
        This tells us what features the model expects as input.
        """
        try:
            # Updated to match your exact filename
            features_path = os.path.join(self.models_path, 'numeric_feature_names(jason file).json')
            with open(features_path, 'r') as f:
                self.numeric_features = json.load(f)
            print("Feature names loaded successfully!")
            print(f"Loaded {len(self.numeric_features)} features: {self.numeric_features}")
            return True
        except Exception as e:
            print(f"Error loading feature names: {e}")
            return False

    def load_all_components(self):
        """
        Load all model components (model, scaler, vectorizer, features).
        Returns True if essential components loaded successfully.
        """
        success = True

        # Load essential components
        success &= self.load_model()
        success &= self.load_feature_names()

        # Load optional components (don't fail if missing)
        scaler_success = self.load_scaler()
        tfidf_success = self.load_tfidf_vectorizer()

        if success:
            print("✅ Essential model components loaded successfully!")
            if not scaler_success:
                print("⚠️  Scaler not loaded - will use raw features")
            if not tfidf_success:
                print("⚠️  TF-IDF vectorizer not loaded - may affect text features")
        else:
            print("❌ Essential components failed to load!")

        return success

    def prepare_input_data(self, input_dict):
        """
        Prepare input data for prediction by creating a feature vector.

        Args:
            input_dict (dict): Dictionary with feature names as keys and values

        Returns:
            numpy.ndarray: Prepared feature vector ready for prediction
        """
        try:
            # Create a feature vector with zeros
            feature_vector = np.zeros(len(self.numeric_features))

            # Fill in the values from input_dict
            for i, feature_name in enumerate(self.numeric_features):
                if feature_name in input_dict:
                    feature_vector[i] = float(input_dict[feature_name])

            # Scale the features using the loaded scaler (if available)
            feature_vector = feature_vector.reshape(1, -1)  # Reshape for single sample

            if self.scaler is not None:
                scaled_features = self.scaler.transform(feature_vector)
                print("Features scaled using loaded scaler")
            else:
                scaled_features = feature_vector
                print("No scaler applied - using raw features")

            return scaled_features

        except Exception as e:
            print(f"Error preparing input data: {e}")
            return None

    def predict(self, input_dict):
        """
        Make a prediction using the loaded model.

        Args:
            input_dict (dict): Dictionary with feature values

        Returns:
            tuple: (prediction, probability) where prediction is 0/1 and probability is confidence
        """
        try:
            # Prepare the input data
            prepared_data = self.prepare_input_data(input_dict)

            if prepared_data is None:
                return None, None

            # Make prediction
            prediction = self.model.predict(prepared_data)[0]

            # Get prediction probability (confidence)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(prepared_data)[0]
                # Get probability of positive class (insider threat)
                threat_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            else:
                # If model doesn't support probability, use decision function or default
                threat_probability = 0.5

            return int(prediction), float(threat_probability)

        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None

    def get_feature_names(self):
        """
        Get the list of feature names that the model expects.

        Returns:
            list: List of feature names
        """
        return self.numeric_features if self.numeric_features else []

    def is_ready(self):
        """
        Check if all model components are loaded and ready for prediction.

        Returns:
            bool: True if essential components are loaded, False otherwise
        """
        # Essential components: model and feature names
        essential_ready = (self.model is not None and
                           self.numeric_features is not None)

        # Optional components: scaler and tfidf_vectorizer
        if essential_ready:
            if self.scaler is None:
                print("Warning: No scaler loaded - using raw features")
            if self.tfidf_vectorizer is None:
                print("Warning: No TF-IDF vectorizer loaded")

        return essential_ready