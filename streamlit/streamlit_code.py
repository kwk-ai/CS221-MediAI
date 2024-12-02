import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import base64
from PIL import Image
import io
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from langchain.schema import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

CHAT_URL = "http://localhost:18001/v1/"
NIM_8B = "meta/llama-3.1-8b-instruct"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize with ImageNet mean and std if used during training
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load models with caching
@st.cache_resource
def load_image_model():
    # Paths
    model_path = 'stomach_classification_model_10_eopch_weight1.pth'  # Update with your model path
    class_to_idx_path = 'class_to_idx.pkl'  # Path to the saved class-to-index mapping

    # Load the class_to_idx mapping
    with open(class_to_idx_path, 'rb') as f:
        class_to_idx = pickle.load(f)
    # Create idx_to_class mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Initialize the model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, len(class_names))
    )
    model = model.to(device)

    # Load the saved model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, class_names

@st.cache_resource
def load_tabular_model():
    # Load the XGBoost model
    xgb_model = XGBClassifier()
    xgb_model.load_model('xgb_model.json')  # Ensure you've saved your model in XGBoost's native format
    return xgb_model

# Load the models
model, class_names = load_image_model()
tabular_model = load_tabular_model()

# Define classify_image function
def classify_image(image_base64: str) -> str:
    """
    Classify a medical image into one of the predefined categories.
    """
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    input_tensor = val_test_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]
    return predicted_class

def predict_severity(df) -> str:
    """
    Predict the severity score based on patient data.
    """
    # Preprocess the data
    print("Original sample data:")
    print(df)
    
    # Load the pre-trained scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Transform the data using the loaded scaler
    X = scaler.transform(df)
    print("Transformed inference sample:")
    print(X)

    # Make predictions
    severity_scores = tabular_model.predict(X)
    print(f"Tabular data insight is {severity_scores}")
    key_map = {
        0: "Cancer",
        1: "Low-grade gastric epithelial dysplasia",
        2: "Intestinal metaplasia",
        3: "Atrophic gastritis",
        4: "Non-atrophic gastritis"
    }
    
    # Map the prediction to the corresponding label
    predicted_labels = [key_map[score] for score in severity_scores]
    return predicted_labels[0]  # Assuming you want the first prediction

# Initialize session state for chat history and data results
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'predicted_class' not in st.session_state:
    st.session_state['predicted_class'] = None
if 'tabular_data_insight' not in st.session_state:
    st.session_state['tabular_data_insight'] = None

st.title('AI-Powered Medical Assistant')

# Sidebar for data upload
st.sidebar.header("Upload Data")
uploaded_image = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
uploaded_file = st.sidebar.file_uploader("Upload Tabular Data (CSV)", type=['csv'])

# Handle Image Upload
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    image_data = uploaded_image.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    predicted_class = classify_image(image_base64)
    # Store the result in session state
    st.session_state['predicted_class'] = predicted_class

# Handle Tabular Data Upload
if uploaded_file:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    
    # Check if there is at least one row; otherwise, raise an error
    if df.shape[0] < 1:
        st.error("The uploaded file does not contain any data.")
    else:
        # Select the first row for prediction
        df_selected = df.iloc[0:1]
        st.dataframe(df_selected, hide_index=True)
        print(df_selected)
        
        # Call the prediction function and display the severity score
        severity_score = predict_severity(df_selected)
        print(f"Severity is: {severity_score}")
        
        # Store the result in session state
        st.session_state['tabular_data_insight'] = severity_score

# Display Chat History
if 'messages' in st.session_state:
    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

# Define function to generate a general response
def generate_general_response(prompt_text):
    # Use the LLM to generate a general response
    model = ChatNVIDIA(
        base_url=CHAT_URL,
        model=NIM_8B,
        max_tokens=256
    )
    system_prompt = "You are a helpful medical assistant. Answer the user's question in a polite manner and offer medical help."
    user_prompt = prompt_text

    messages = [
        HumanMessage(content=system_prompt, role="system"),
        HumanMessage(content=user_prompt, role="user")
    ]

    # Return the model and messages
    return model, messages

def talk_with_data(prompt_text):
    # Retrieve the image classification result and severity prediction from session state
    predicted_class = st.session_state.get('predicted_class', None)
    tabular_data_insight = st.session_state.get('tabular_data_insight', None)
    
    # Build the context based on the available data
    data_context = ""
    if predicted_class:
        data_context += f"Image diagnosis result: {predicted_class}.\n"
    if tabular_data_insight is not None:
        data_context += f"Tabular Data insight: {tabular_data_insight}.\n"
    
    # Use the LLM to generate a response with the data context
    model = ChatNVIDIA(
        base_url=CHAT_URL,
        model=NIM_8B,
        max_tokens=512
    )
    system_prompt = "You are a medical assistant that helps with the first level of triaging and helps patients understand how you came to this possible diagnosis, and helps them understand their symptoms. If presented with the data, first start with saying 'Based on the evidence given, it is likely that you have ...', then give feedback based on the user's disease, both images or textual data. Do not give life-style recommendations, but instead give the next step for triaging, for example, recommend the patient to seek medical help in a certain department for certain in-depth test and analysis"
    
    if data_context:
        system_prompt += "\n\nHere is the patient's data:\n" + data_context
    
    user_prompt = prompt_text
    
    messages = [
        HumanMessage(content=system_prompt, role="system"),
        HumanMessage(content=user_prompt, role="user")
    ]

    # Return the model and messages
    return model, messages

# Chat Input at the bottom
chat_prompt = st.chat_input("Type your message here...")

# Handle General Chat
if chat_prompt:
    # Add user message to chat history
    st.session_state['messages'].append({"role": "user", "content": chat_prompt})

    # Display the user's message
    with st.chat_message("user"):
        st.markdown(chat_prompt)

    if st.session_state.get('predicted_class') or st.session_state.get('tabular_data_insight'):
        # Use agent for analysis when data is present
        try:
            model, messages = talk_with_data(chat_prompt)
            st.session_state['messages'].append({"role": "assistant", "content": ""})
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in model.stream(messages):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response)
                st.session_state['messages'][-1]['content'] = full_response
        except Exception as e:
            response = f"An error occurred: {e}"
            st.session_state['messages'].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
    else:
        # General chat if no data is uploaded
        try:
            model, messages = generate_general_response(chat_prompt)
            st.session_state['messages'].append({"role": "assistant", "content": ""})
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in model.stream(messages):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response)
                st.session_state['messages'][-1]['content'] = full_response
        except Exception as e:
            response = f"An error occurred: {e}"
            st.session_state['messages'].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)