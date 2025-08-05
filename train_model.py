import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score # Import accuracy_score
import joblib
import os

def train_and_save_model(file_path='global_sales_data.xlsx', sheet_name='global_sales_data'):
    """
    Loads sales data, preprocesses it, trains a RandomForestClassifier to predict
    'Sub-Category', and saves the trained model and label encoders.
    """
    print(f"🚀 Starting model training process with data from: {file_path}")

    try:
        # Step 1: Load the data
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print("✅ Data loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: The file '{file_path}' was not found. Please ensure it's in the same directory.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Step 2: Preprocess Data
    # Convert 'Order Date' to datetime and extract Month and Year
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Month'] = df['Order Date'].dt.month
    df['Year'] = df['Order Date'].dt.year

    # Select features and target
    features = ['Region', 'Salesperson', 'Month', 'Year']
    target = 'Sub-Category'

    # Check if all required columns exist
    if not all(col in df.columns for col in features + [target]):
        missing_cols = [col for col in features + [target] if col not in df.columns]
        print(f"❌ Error: Missing required columns in the dataset: {', '.join(missing_cols)}")
        return

    X = df[features]
    y = df[target]

    # Initialize LabelEncoders for categorical features and the target
    le_region = LabelEncoder()
    le_salesperson = LabelEncoder()
    le_target = LabelEncoder()

    # Fit and transform categorical features
    X['Region'] = le_region.fit_transform(X['Region'])
    X['Salesperson'] = le_salesperson.fit_transform(X['Salesperson'])

    # Fit and transform the target variable
    y_encoded = le_target.fit_transform(y)
    print("✅ Data preprocessing complete.")

    # Step 3: Train the Model
    # Split data into training and testing sets for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    print(f"📊 Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

    # Initialize and train a RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train) # Train on the training data
    print("✅ RandomForestClassifier trained successfully.")

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy on Test Set: {accuracy:.2f}")

    # Step 4: Save the trained model and encoders
    joblib.dump(classifier, 'sub_category_predictor.pkl')
    joblib.dump(le_region, 'encoder_region.pkl')
    joblib.dump(le_salesperson, 'encoder_salesperson.pkl')
    joblib.dump(le_target, 'encoder_target.pkl')
    print("✅ Model and encoders saved as '.pkl' files.")
    print("You can now run your chatbot script, and it will use these trained files.")

if __name__ == "__main__":
    # Ensure your global_sales_data.xlsx file is in the same directory as this script.
    train_and_save_model()
