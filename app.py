import pandas as pd
import joblib
import json
import time
import os
import google.generativeai as genai

# --- API Setup (IMPORTANT) ---
# Configure the Google Generative AI with your API key.
# It's recommended to set this as an environment variable for security.
# For local execution, you can directly paste your key here.
# For deployment or sharing, use environment variables as shown in the commented line.
api_key = "AIzaSyC5IqTq8FBMMbNeDJyZM83b5ettlI-nias" # Paste your actual API key here
# api_key = os.getenv("GOOGLE_API_KEY", "") # Recommended: Get API key from environment variable named GOOGLE_API_KEY

genai.configure(api_key=api_key)

# Initialize the generative model for natural language understanding
generative_model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')

# --- Loading Model, Encoders, and Data ---
# These files must be in the same directory as this script.
try:
    # Load the trained machine learning model
    model_predictor = joblib.load('sub_category_predictor.pkl')
    # Load the label encoder for 'Region'
    le_region = joblib.load('encoder_region.pkl')
    # Load the label encoder for 'Salesperson'
    le_salesperson = joblib.load('encoder_salesperson.pkl')
    # Load the label encoder for the target variable 'Sub-Category'
    le_target = joblib.load('encoder_target.pkl')
    print("✅ Model and encoders loaded successfully from .pkl files.")
except FileNotFoundError:
    print("❌ Error: One or more model files not found.")
    print("Please ensure 'sub_category_predictor.pkl', 'encoder_region.pkl', 'encoder_salesperson.pkl', and 'encoder_target.pkl' are in the same directory as this script.")
    print("You need to run the 'train_sales_model.py' script first to generate these files.")
    exit("Exiting: Missing model files.")
except Exception as e:
    print(f"❌ An unexpected error occurred while loading model files: {e}")
    exit("Exiting: Model loading failed.")

# Load the main sales data for summary calculations
try:
    sales_df = pd.read_excel('global_sales_data.xlsx', sheet_name='global_sales_data')
    print("✅ Sales data loaded successfully from 'global_sales_data.xlsx'.")
except FileNotFoundError:
    print("❌ Error: 'global_sales_data.xlsx' not found.")
    print("Please ensure 'global_sales_data.xlsx' is in the same directory as this script.")
    exit("Exiting: Missing sales data file.")
except Exception as e:
    print(f"❌ An unexpected error occurred while loading sales data: {e}")
    exit("Exiting: Sales data loading failed.")


# --- Data Analysis Functions ---
def get_user_prediction_input():
    """
    Prompts the user for details needed for a sub-category prediction.
    It guides the user to provide valid inputs based on the loaded encoders.
    """
    print("\n📝 Please provide some details for the prediction.")

    # Get Region input from user
    regions = list(le_region.classes_)
    print(f"Available Regions: {', '.join(regions)}")
    while True:
        region = input("Enter Region: ").strip()
        if region in regions:
            break
        print("⚠️ Invalid region. Please choose from the list.")

    # Get Salesperson input from user
    salespeople = list(le_salesperson.classes_)
    print(f"Available Salespeople: {', '.join(salespeople)}")
    while True:
        salesperson = input("Enter Salesperson Name: ").strip()
        if salesperson in salespeople:
            break
        print("⚠️ Invalid salesperson. Please choose from the list.")

    # Get Month input from user with validation
    while True:
        try:
            month = int(input("Enter Month (1-12): "))
            if 1 <= month <= 12:
                break
            else:
                print("⚠️ Invalid month. Must be between 1 and 12.")
        except ValueError:
            print("⚠️ Invalid input. Please enter a number for the month.")

    # Get Year input from user with validation
    while True:
        try:
            year = int(input("Enter Year (e.g., 2023): "))
            if year > 0: # Basic validation for a positive year
                break
            else:
                print("⚠️ Invalid year. Please enter a valid year.")
        except ValueError:
            print("⚠️ Invalid input. Please enter a number for the year.")

    return {
        'Region': region,
        'Salesperson': salesperson,
        'Month': month,
        'Year': year
    }

def predict_sub_category():
    """
    Collects user input, transforms it using loaded encoders, and makes a prediction
    using the loaded machine learning model.
    """
    user_input = get_user_prediction_input()
    if user_input is None:
        print("🤖 Prediction cancelled.")
        return

    # Transform categorical input using the loaded encoders
    # Ensure the input is a list as .transform expects an array-like input
    encoded_region = le_region.transform([user_input['Region']])[0]
    encoded_salesperson = le_salesperson.transform([user_input['Salesperson']])[0]

    # Create a pandas DataFrame for the model prediction
    # The column names must match the features the model was trained on
    sample = pd.DataFrame([{
        'Region': encoded_region,
        'Salesperson': encoded_salesperson,
        'Month': user_input['Month'],
        'Year': user_input['Year']
    }])

    # Make a prediction using the loaded model
    pred_encoded = model_predictor.predict(sample)
    # Inverse transform the numerical prediction back to the original sub-category label
    pred_label = le_target.inverse_transform(pred_encoded)[0]

    print(f"\n🤖 Based on your input, the predicted top sub-category is: **{pred_label}**")

def get_sales_summary(summary_type):
    """
    Calculates and prints summary statistics (total, max, min) for 'Sales'.
    Args:
        summary_type (str): The type of summary to perform ('total', 'max', 'min').
    """
    if 'Sales' not in sales_df.columns:
        print("❌ Error: 'Sales' column not found in the data.")
        return

    if summary_type == 'total':
        total_sales = sales_df['Sales'].sum()
        print(f"\n📈 Total Sales: **${total_sales:,.2f}**")
    elif summary_type == 'max':
        max_sales = sales_df['Sales'].max()
        print(f"\n⬆️ Maximum Sales: **${max_sales:,.2f}**")
    elif summary_type == 'min':
        min_sales = sales_df['Sales'].min()
        print(f"\n⬇️ Minimum Sales: **${min_sales:,.2f}**")
    else:
        print("🤖 I can only provide 'total', 'max', or 'min' sales summaries.")


def list_available_options():
    """
    Informs the user about the chatbot's capabilities and available data options
    (regions and salespeople).
    """
    regions = ', '.join(le_region.classes_)
    salespeople = ', '.join(le_salesperson.classes_)
    print(f"🤖 I can help you predict the top-performing sub-category.")
    print(f"Available regions: {regions}")
    print(f"Available salespeople: {salespeople}")
    print(f"I can also tell you about total, maximum, or minimum sales.")
    print(f"Just ask me a question like: 'What is the best sub-category for Jane Doe in the West in March?' or 'What are the total sales?'")

# --- Main Chatbot Loop ---
def chatbot():
    """
    The main conversational loop of the chatbot. It takes user input,
    uses a generative AI model to understand the intent, and executes
    the corresponding function.
    """
    print("✨ Welcome to the Sales Insight Chatbot!")
    print("How can I help you today? (Type 'exit' to quit)")

    # Map commands identified by the generative model to local functions
    commands = {
        'predict_sub_category': predict_sub_category,
        'list_options': list_available_options,
        'get_sales_summary': get_sales_summary # New command for sales summaries
    }

    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in ['exit', 'quit', 'q', 'bye']:
            print("👋 Goodbye! Have a great day!")
            break

        if not user_text: # Handle empty input
            continue

        # Construct the prompt for the generative AI model
        prompt = f"""
            You are a command-line chatbot that helps a user predict sales sub-categories and provide sales summaries. Your task is to determine which function to execute based on the user's natural language request. You should respond with a JSON object.

            Available commands:
            - `predict_sub_category`: To predict the top sub-category. Trigger this for questions about prediction, forecasts, or best sub-categories.
            - `get_sales_summary`: To get total, maximum, or minimum sales. Trigger this for questions like "total sales", "max sales", "min sales". If this command is chosen, also include a 'summary_type' field with value 'total', 'max', or 'min'.
            - `list_options`: To list the available regions and salespeople and what the bot can do.
            - `unknown`: Use this if the user's request doesn't match any of the above commands.

            User's request: "{user_text}"

            Respond with a JSON object containing a "command" field and a "response" field. If the command is `get_sales_summary`, also include a "summary_type" field. The "response" field should be a conversational message to the user.
            
            Example response for 'What is the top sub-category for Jane Doe in the West?':
            {{
                "command": "predict_sub_category",
                "response": "I can help with that! Let's get a few more details."
            }}
            
            Example response for 'What are the total sales?':
            {{
                "command": "get_sales_summary",
                "response": "Calculating total sales...",
                "summary_type": "total"
            }}

            Example response for 'What is the maximum sales?':
            {{
                "command": "get_sales_summary",
                "response": "Finding the maximum sales...",
                "summary_type": "max"
            }}
            
            Example response for 'What can you do?':
            {{
                "command": "list_options",
                "response": "I can predict the top-performing sub-category based on region, salesperson, month, and year, and provide sales summaries."
            }}
            
            Example response for 'Hello':
            {{
                "command": "unknown",
                "response": "Hello! I can help you with sales predictions and summaries. How can I assist you?"
            }}
            
            Please provide a JSON object for the user's request.
        """

        try:
            # Implement exponential backoff for API calls to handle rate limits
            retries = 0
            while retries < 5: # Max 5 retries
                try:
                    # Generate content using the generative AI model
                    response = generative_model.generate_content(prompt)
                    
                    # --- DEBUGGING STEP: Print raw response text ---
                    raw_response_text = response.text.strip()
                    print(f"DEBUG: Raw API response: '{raw_response_text}'")
                    # --- END DEBUGGING STEP ---

                    # Parse the JSON response from the model
                    api_response = json.loads(raw_response_text)
                    command = api_response.get('command')
                    bot_response = api_response.get('response')
                    summary_type = api_response.get('summary_type') # Get summary_type if present

                    print(f"🤖 {bot_response}")

                    # Execute the identified command
                    if command and command in commands:
                        if command == 'get_sales_summary' and summary_type:
                            commands[command](summary_type) # Pass summary_type argument
                        else:
                            commands[command]()
                    else:
                        # If command is 'unknown' or not recognized, list options
                        list_available_options()
                    break # Break from the retry loop on success
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"🤖 Sorry, I had trouble processing that request (JSON error: {e}). Retrying...")
                    retries += 1
                    time.sleep(2 ** retries) # Exponential backoff
                except Exception as e:
                    print(f"🤖 An unexpected error occurred during API call: {e}")
                    break # Break on other unexpected errors
            if retries == 5:
                print("🤖 I'm having persistent issues connecting to the AI. Please try again later.")
        except Exception as e:
            print(f"🤖 An error occurred before the API call could be made: {e}")

if __name__ == "__main__":
    chatbot()
