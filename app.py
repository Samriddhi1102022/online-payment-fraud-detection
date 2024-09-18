import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('fraud_detection_model.pkl')

# Function to make predictions
def predict_fraud(data):
    # Get probability of fraud
    prob = model.predict_proba(data)[:, 1]
    # Convert probability to binary classification
    return prob[0] > 0.5  # Adjust the threshold if needed

# Streamlit UI
st.title('Online Payment Fraud Detection')

# User inputs
step = st.number_input('Step', min_value=0)
type_transfer = st.checkbox('Type: TRANSFER')
type_cash_out = st.checkbox('Type: CASH_OUT')
type_deposit = st.checkbox('Type: DEPOSIT')
type_payment = st.checkbox('Type: PAYMENT')

# Convert type to one-hot encoding
type_transfer_val = 1 if type_transfer else 0
type_cash_out_val = 1 if type_cash_out else 0
type_deposit_val = 1 if type_deposit else 0
type_payment_val = 1 if type_payment else 0

amount = st.number_input('Transaction Amount', min_value=0.0)
old_balance_org = st.number_input('Sender Old Balance', min_value=0.0)
new_balance_org = st.number_input('Sender New Balance', min_value=0.0)
old_balance_dest = st.number_input('Receiver Old Balance', min_value=0.0)
new_balance_dest = st.number_input('Receiver New Balance', min_value=0.0)

# Create a DataFrame for prediction with correct feature names
input_data = pd.DataFrame({
    'step': [step],
    'amount': [amount],
    'oldbalanceOrg': [old_balance_org],
    'newbalanceOrig': [new_balance_org],
    'oldbalanceDest': [old_balance_dest],
    'newbalanceDest': [new_balance_dest],
    'CASH_OUT': [type_cash_out_val],
    'DEBIT': [0],  # Assuming 'DEBIT' is not used in this example
    'PAYMENT': [type_payment_val],
    'TRANSFER': [type_transfer_val]
})

# Predict
if st.button('Predict'):
    is_fraud = predict_fraud(input_data)
    if is_fraud:
        st.write('Transaction is Fraud.')
    else:
        st.write('Transaction is Not Fraud.')