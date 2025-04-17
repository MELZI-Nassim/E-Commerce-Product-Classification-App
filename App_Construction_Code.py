### Importing the necessary libraries ###
import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import nltk

### Download NLTK stopwords once ###
nltk.download('stopwords')

### Constants ###
TEMPLATE_PATH = "Template.xlsx"
UPLOAD_SIZE_LIMIT_MB = 5120  # 5 Go = 5120 Mo

### Set the working directory to the specified path ###
os.chdir("C:\\Users\\Fateh-Nassim MELZI\\Documents\\AI_Projects\\E-Commerce_Product_Classification_Project\\App_Construction")

### Set page configuration ###
st.set_page_config(page_title="E-Commerce Product Classification", page_icon="🛒", layout="centered")

### Text preprocessing function ###
def text_preprocessing(text: str) -> str:
    ### Substituting punctuation by space except underscore ###
    text = re.sub(r'\W', " ", text, flags=re.UNICODE)       
    ### Substituting underscore by space ###
    text = re.sub(r'_', " ", text, flags=re.UNICODE) 
    ### Substituting digits by space ###
    text = re.sub(r'\d+', " ", text, flags=re.UNICODE)
    ### Converting text to lowercase ###
    text = text.lower()
    ### Substituting single letter by space ###
    text = re.sub(r"\b[a-zA-Z]\b", " ", text, flags=re.UNICODE)
    ### Split the text into tokens ###
    text = text.split()
    ### Remove all French stopwords ###
    stop_words = set(stopwords.words('french'))
    text = [word for word in text if word not in stop_words]
    ### Stemming the words ###
    stemmer = FrenchStemmer()
    text = [stemmer.stem(word) for word in text] 
    text = " ".join(text)
    return text


### Function to classify products based on their designations ###
def classify_products(model, vectorizer, descriptions: pd.Series) -> pd.DataFrame:
    processed_descriptions = descriptions.apply(text_preprocessing)
    dtm = vectorizer.transform(processed_descriptions)
    X = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Check if all rows in X are zeros
    unknown_class = "Je ne sais pas, il faut rajouter plus de mots clés"
    y_pred = []
    for i in range(X.shape[0]):
        if X.iloc[i].sum() == 0:
            y_pred.append(unknown_class)
        else:
            y_pred.append(model.predict(X.iloc[i].values.reshape(1, -1))[0])
    
    results = pd.DataFrame({
        'Description': descriptions,
        'Predicted Classes': y_pred
    })
    
    return results

# Load the pre-prepared Excel template
def load_template(file_path: str) -> bytes:
    with open(file_path, "rb") as file:
        return file.read()


### Cache the model loading function to avoid reloading on every interaction ###
@st.cache_resource
def load_model():
    try:
        return joblib.load("Model.pkl.xz")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'Model.pkl.xz' is in the correct directory.")
        return None

### Cache the encoders loading function to avoid reloading on every interaction ###
@st.cache_resource
def load_vectorizer():
    try:
        return joblib.load("Vectorizer.pkl.xz")
    except FileNotFoundError:
        st.error("Vectorizer file not found. Please ensure 'Vectorizer.pkl.xz' is in the correct directory.")
        return None


### Load the trained model and encoders ###
model = load_model()
vectorizer = load_vectorizer()

### Export the classification results into excel format ###
def to_excel_data(data: pd.DataFrame, index: bool = False) -> bytes:
    template_io = BytesIO()
    data.to_excel(template_io, index=index)
    return template_io.getvalue()

# Streamlit app title
st.header('🛒 E-Commerce Product Classification')

# Add explanation and image to the sidebar
st.sidebar.title("ℹ️ About:")
st.sidebar.write("""
This AI-powered application 🤖 classifies products into 20 distinct categories based on their textual descriptions.
By leveraging advanced machine learning algorithms 🧠, it delivers precise classifications to support informed decision-making 📊.
You can classify either a single product or multiple products at once by uploading an Excel file. 
""")

st.sidebar.image("E-Commerce_Image.jpg", caption="", use_container_width=True)

# Create a single radio button group for prediction mode
prediction_mode = st.radio("Choose the classification mode:", ("One product", "Several products"), index=0, horizontal=True)

# If the classification mode is "One product"
if prediction_mode == "One product":
    # Create a text input for the product description
    product_description = st.text_area("Enter the product description (in French):", max_chars=500, height=68)
    
    # Create a button to classify the product
    if st.button("Classify Product"):
        if not product_description:
            st.error("Please enter a product description.")
        else:
            # Perform classification (assuming model and vectorizer are already loaded)
            data = pd.Series([product_description])
            result = classify_products(model, vectorizer, data)
            
            # Display the result
            predicted_classes = result['Predicted Classes'].iloc[0]
            if predicted_classes == "Je ne sais pas, il faut rajouter plus de mots clés":
                st.error("Je ne sais pas, il faut rajouter plus de mots clés")
            else:
                st.success(f"Predicted Category: {predicted_classes}")

# If the prediction mode is "Several products"

else:
    cols = st.columns(2)
    template = load_template(TEMPLATE_PATH)
    cols[0].download_button('Download a template', template, file_name='template.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', help='Download an Excel file template')
    
    # Add an information note
    st.info("""
    ⚠️ **Note:** please ensure that the header in your uploaded file matches exactly as specified in the template. This will help avoid errors during the classification process.
    """)
    
    # File uploader for Excel file with size limit information
    uploaded_file = st.file_uploader(
        label=f"Select a file with the products to classify",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=False,
        key="pred_parts",
        help="Click on browser to upload your file (Max size: 5 GB)"
    )
    if uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)
        
        # Check if the required column is present
        if 'Description' not in df.columns:
            st.error("The uploaded file must contain a 'Description' column.")
        else:
            # Perform classification
            descriptions = df['Description']
            results = classify_products(model, vectorizer, descriptions)
            
            # Display the results
            st.write(results)
            
            # Provide an option to download the results as an Excel file
            excel_data = to_excel_data(results)
            st.download_button(label="Download Results", data=excel_data, file_name="classification_results.xlsx")