# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Create the dataset
print("=" * 50)
print("CREDIT RISK MODEL - RBI Style")
print("=" * 50)

# Creating data for 20 customers
data = {
    'Age': [28, 45, 35, 52, 23, 41, 38, 29, 55, 33, 
            26, 48, 31, 44, 27, 50, 36, 42, 30, 39],
    
    'Monthly_Income': [35000, 85000, 55000, 120000, 22000, 70000, 48000, 38000, 95000, 52000,
                       28000, 78000, 45000, 88000, 32000, 110000, 62000, 75000, 42000, 58000],
    
    'Employment_Type': [0, 0, 1, 0, 1, 0, 1, 0, 0, 1,
                        0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    
    'Loan_Amount': [200000, 500000, 350000, 800000, 150000, 450000, 280000, 220000, 600000, 320000,
                    180000, 520000, 300000, 550000, 190000, 750000, 420000, 480000, 250000, 380000],
    
    'EMI_Amount': [8000, 18000, 14000, 28000, 7000, 16000, 12000, 9000, 21000, 13000,
                   7500, 19000, 12500, 20000, 8500, 26000, 15500, 17500, 10500, 14500],
    
    'Existing_Loans': [1, 0, 2, 1, 2, 1, 2, 1, 0, 2,
                       2, 1, 2, 0, 3, 1, 1, 1, 2, 2],
    
    'Missed_Payments': [0, 0, 1, 0, 3, 0, 2, 1, 0, 1,
                        4, 0, 2, 0, 5, 0, 1, 0, 3, 1],
    
    'Credit_Utilization': [25, 15, 45, 20, 78, 30, 65, 38, 18, 52,
                           85, 22, 58, 16, 92, 24, 42, 28, 72, 48],
    
    'Risk_Category': [0, 0, 1, 0, 2, 0, 1, 1, 0, 1,
                      2, 0, 1, 0, 2, 0, 1, 0, 2, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

print("\nOur Credit Risk Dataset:")
print(df.head(10))
print(f"\nTotal Customers: {len(df)}")

# Step 2: Prepare data for machine learning
print("\n" + "=" * 50)
print("STEP 2: Training the AI Model")
print("=" * 50)

# Separate features and target
X = df.drop('Risk_Category', axis=1)
y = df['Risk_Category']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data: {len(X_train)} customers")
print(f"Testing data: {len(X_test)} customers")

# Step 3: Train the model
print("\nTraining the Decision Tree model...")
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)
print("Model trained successfully!")

# Step 4: Test the model
print("\n" + "=" * 50)
print("STEP 3: Testing Model Accuracy")
print("=" * 50)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"\nModel Accuracy: {accuracy:.2f}%")
print("\nThis means the model correctly predicts credit risk")
print(f"{accuracy:.2f}% of the time based on the patterns it learned!")

# Step 5: Predict for a NEW customer
print("\n" + "=" * 50)
print("STEP 4: Predicting for a NEW Customer")
print("=" * 50)

new_customer = {
    'Age': 32,
    'Monthly_Income': 45000,
    'Employment_Type': 0,
    'Loan_Amount': 250000,
    'EMI_Amount': 10000,
    'Existing_Loans': 1,
    'Missed_Payments': 2,
    'Credit_Utilization': 65
}

print("\nNew Customer Details:")
print(f"   Age: {new_customer['Age']} years")
print(f"   Monthly Income: Rs.{new_customer['Monthly_Income']:,}")
print(f"   Employment: {'Salaried' if new_customer['Employment_Type'] == 0 else 'Self-Employed'}")
print(f"   Loan Amount: Rs.{new_customer['Loan_Amount']:,}")
print(f"   EMI Amount: Rs.{new_customer['EMI_Amount']:,}")
print(f"   Existing Loans: {new_customer['Existing_Loans']}")
print(f"   Missed Payments: {new_customer['Missed_Payments']}")
print(f"   Credit Utilization: {new_customer['Credit_Utilization']}%")

new_customer_df = pd.DataFrame([new_customer])
prediction = model.predict(new_customer_df)[0]

print("\n" + "=" * 50)
print("PREDICTION RESULT")
print("=" * 50)

risk_labels = {0: "LOW RISK", 1: "MEDIUM RISK", 2: "HIGH RISK"}
print(f"\nCredit Risk Category: {risk_labels[prediction]}")

if prediction == 0:
    print("RECOMMENDATION: Approve loan - Customer is low risk")
elif prediction == 1:
    print("RECOMMENDATION: Proceed with caution - Medium risk customer")
else:
    print("RECOMMENDATION: Reject loan - Customer is high risk")

print("\n" + "=" * 50)
print("Credit Risk Model Complete!")
print("=" * 50)