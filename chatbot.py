"""
AI Chatbot for Credit Card Fraud Detection
===========================================
Uses Google's Generative AI to provide intelligent assistance
about fraud detection, transactions, and model explanations.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("Warning: GOOGLE_API_KEY not found in .env file")


def initialize_chatbot():
    """
    Initialize the chatbot with a system prompt.
    
    Returns:
    --------
    model : GenerativeModel
        The configured chatbot model
    """
    if not GOOGLE_API_KEY:
        return None
    
    try:
        # System prompt to make the chatbot helpful for fraud detection
        system_instruction = """You are an AI assistant specialized in credit card fraud detection. 
Your role is to help users understand:
1. How fraud detection models work
2. What transaction features are important
3. How to interpret fraud predictions
4. Best practices for fraud prevention
5. Explain model predictions and probabilities

Be helpful, clear, and educational. If asked about specific transactions, 
provide insights based on general fraud detection principles."""
        
        # Use Gemini Pro model
        model = genai.GenerativeModel(
            model_name='gemini-pro',
            system_instruction=system_instruction
        )
        
        return model
        
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return None


def get_chatbot_response(user_message, conversation_history=None):
    """
    Get a response from the chatbot.
    
    Parameters:
    -----------
    user_message : str
        The user's message/question
    conversation_history : list, optional
        Previous conversation messages for context
        
    Returns:
    --------
    response : str
        The chatbot's response
    error : str or None
        Error message if something went wrong
    """
    if not GOOGLE_API_KEY:
        return None, "API key not configured. Please check your .env file."
    
    try:
        model = initialize_chatbot()
        if not model:
            return None, "Failed to initialize chatbot model."
        
        # Build the conversation
        if conversation_history:
            # Format history for the model
            messages = conversation_history + [user_message]
        else:
            messages = [user_message]
        
        # Generate response
        response = model.generate_content(messages)
        
        return response.text, None
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(error_msg)
        return None, error_msg


def get_fraud_explanation(prediction_result):
    """
    Get an AI-generated explanation of a fraud prediction.
    
    Parameters:
    -----------
    prediction_result : dict
        Dictionary containing prediction results with keys:
        - is_fraud: bool
        - fraud_probability: float
        - not_fraud_probability: float
        - confidence: float
        
    Returns:
    --------
    explanation : str
        AI-generated explanation of the prediction
    """
    if not GOOGLE_API_KEY:
        return "Chatbot not available. API key not configured."
    
    try:
        model = initialize_chatbot()
        if not model:
            return "Chatbot model not available."
        
        # Create a detailed prompt for explanation
        prompt = f"""Explain this fraud detection result in simple terms:

Prediction: {'FRAUD DETECTED' if prediction_result.get('is_fraud') else 'Transaction appears SAFE'}
Fraud Probability: {prediction_result.get('fraud_probability', 0):.2%}
Not Fraud Probability: {prediction_result.get('not_fraud_probability', 0):.2%}
Confidence: {prediction_result.get('confidence', 0):.2%}

Provide:
1. What this prediction means
2. Why the model made this prediction
3. What the user should do next
4. Any important considerations

Keep it concise (2-3 sentences) and user-friendly."""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Unable to generate explanation: {str(e)}"


if __name__ == "__main__":
    # Test the chatbot
    print("Testing Chatbot...")
    print("=" * 60)
    
    if not GOOGLE_API_KEY:
        print("❌ API key not found. Please check your .env file.")
    else:
        print("✓ API key loaded")
        
        # Test basic response
        response, error = get_chatbot_response(
            "What are the most important features for detecting credit card fraud?"
        )
        
        if error:
            print(f"❌ Error: {error}")
        else:
            print("✓ Chatbot response:")
            print(response)
            print("=" * 60)

