"""
File Organization Script for Insider Threat Detection Flask App
Run this script to organize your model files into the correct Flask structure
"""

import os
import shutil


def create_flask_structure():
    """
    Create the required directory structure for the Flask app
    """
    directories = [
        'insider_threat_detection_app',
        'insider_threat_detection_app/models',
        'insider_threat_detection_app/static',
        'insider_threat_detection_app/templates',
        'insider_threat_detection_app/utils'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def copy_model_files():
    """
    Copy your model files from the original location to Flask app structure
    Update the source paths to match your actual file locations
    """

    # Your original model directory path
    source_dir = r"C:\Users\satya\OneDrive\Desktop\projects\insider threat detection model\models"

    # Destination directory in Flask app
    dest_dir = "insider_threat_detection_app/models"

    # File mappings: (source_filename, destination_filename)
    file_mappings = [
        ("insiderthreat model(joblib file).joblib", "insiderthreat model(joblib file).joblib"),
        ("tfidf_vectorizer(joblib).joblib", "tfidf_vectorizer(joblib).joblib"),
        ("numeric_feature_names(jason file).json", "numeric_feature_names(jason file).json"),
        # Add your scaler file here - update with the actual filename
        # ("your_scaler_filename.joblib", "scaler(joblib).joblib"),
    ]

    print(f"Copying files from: {source_dir}")
    print(f"To: {dest_dir}")

    for source_file, dest_file in file_mappings:
        source_path = os.path.join(source_dir, source_file)
        dest_path = os.path.join(dest_dir, dest_file)

        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, dest_path)
                print(f"✅ Copied: {source_file} → {dest_file}")
            except Exception as e:
                print(f"❌ Error copying {source_file}: {e}")
        else:
            print(f"⚠️  File not found: {source_path}")


def list_files_in_source():
    """
    List all files in your original models directory to help identify the scaler file
    """
    source_dir = r"C:\Users\satya\OneDrive\Desktop\projects\insider threat detection model\models"

    print(f"\n📁 Files in your models directory ({source_dir}):")
    print("-" * 60)

    try:
        if os.path.exists(source_dir):
            files = os.listdir(source_dir)
            for i, file in enumerate(files, 1):
                file_path = os.path.join(source_dir, file)
                file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                print(f"{i}. {file} ({file_size} bytes)")
        else:
            print("❌ Source directory not found!")
    except Exception as e:
        print(f"❌ Error listing files: {e}")


def main():
    """
    Main function to organize files for Flask app
    """
    print("🚀 Setting up Flask App Structure for Insider Threat Detection")
    print("=" * 60)

    # Step 1: List existing files to identify missing ones
    list_files_in_source()

    # Step 2: Create Flask directory structure
    print(f"\n📂 Creating Flask app directory structure...")
    create_flask_structure()

    # Step 3: Copy model files
    print(f"\n📋 Copying model files...")
    copy_model_files()

    print(f"\n✅ Setup complete!")
    print(f"\nNext steps:")
    print(f"1. If you see any missing files above, update the file_mappings in this script")
    print(f"2. Navigate to 'insider_threat_detection_app' directory")
    print(f"3. Create the Python files (app.py, utils/insider_threat_model_loader.py, etc.)")
    print(f"4. Create HTML templates and CSS files")
    print(f"5. Run: pip install -r requirements.txt")
    print(f"6. Run: python app.py")


if __name__ == "__main__":
    main()