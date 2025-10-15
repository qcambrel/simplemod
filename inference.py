import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained("./model/final")
tokenizer = BertTokenizer.from_pretrained("./tokenizer/final")

# Utilize CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to evaluation mode
model.eval()

# Create a Streamlit app
st.title("Simplemod")
st.write("Detect fake reviews")

# Provide human or machine generated text for inference
text = st.text_input("Enter a fake review:")

# st.button("Use GPT")
# st.button("Use Claude")

# Tokenize and pad the input
if text:
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt"
    )

    # Move the inputs to the device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_label = torch.argmax(logits, dim=1).item()

    # Provide generated text for inference
    st.write("Fake review" if predicted_label == 1 else "Real review")