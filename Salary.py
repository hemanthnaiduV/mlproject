
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Read the CSV file into a DataFrame
df = pd.read_csv(r"C:\Users\Hemanth\Downloads\Salary Prediction of Data Professions.csv")

# Define the columns to drop
columns_to_drop = ['FIRST NAME', 'LAST NAME']  # Replace with actual column names

# Drop the columns and create a new DataFrame (without inplace modification)
df = df.drop(columns=columns_to_drop, axis=1)

df_1 = df.dropna(subset=['DOJ'])


# Convert 'CURRENT DATE' and 'DOJ' columns to datetime format (assuming MM-DD-YYYY)
def convert_to_datetime(date_str):
  try:
    return pd.to_datetime(date_str, format='%m-%d-%Y')
  except:
    return None

df_1['CURRENT DATE'] = df_1['CURRENT DATE'].apply(convert_to_datetime)
df_1['DOJ'] = df_1['DOJ'].apply(convert_to_datetime)

# Calculate days worked (assuming 'CURRENT DATE' is the current date)
df_1['Days Worked'] = (df_1['CURRENT DATE'] - df_1['DOJ']).dt.days

def fill_and_convert(df, columns_to_fill):
  """Fills null values in specified columns with rounded mean (int) converted to float.

  Args:
      df (pandas.DataFrame): The DataFrame to modify.
      columns_to_fill (list): A list of column names to consider for filling null values.

  Returns:
      pandas.DataFrame: The DataFrame with null values filled and converted.
  """
  for col in columns_to_fill:
    df.loc[df[col].isnull(), col] = df[col].fillna(df[col].mean()).round().astype(int).astype(float)
  return df

# Specify the columns to fill null values (replace with actual columns)
columns_to_fill = ['AGE', 'LEAVES USED', 'LEAVES REMAINING', 'RATINGS']  # Replace with actual column names

# Fill null values with rounded mean (int) converted to float
df = fill_and_convert(df_1, columns_to_fill.copy())

# Define a dictionary for mapping sex to numerical values (0 or 1)
sex_mapping = {'M': 0, 'F': 1}

# Apply the mapping to the 'SEX' column using a lambda function
df_1['SEX_encoded'] = df_1['SEX'].apply(lambda x: sex_mapping[x])

# Define dictionaries for designation and unit encoding
designation_mapping = {
    'Analyst': 1,
    'Associate': 2,
    'Senior Analyst': 3,
    'Senior Manager': 4,
    'Manager': 5,
    'Director': 6
}

unit_mapping = {
    'Finance': 1,
    'Web': 2,
    'IT': 3,
    'Operations': 4,
    'Marketing': 5,
    'Management': 6
}

# Create new columns with encoded values
df_1['designation_encoded'] = df_1['DESIGNATION'].map(designation_mapping)
df_1['unit_encoded'] = df_1['UNIT'].map(unit_mapping)

# Create a new column named "total_leaves" by adding existing columns
df_1['total_leaves'] = df_1['LEAVES USED'] + df_1['LEAVES REMAINING']

df_2=df_1.copy()

from sklearn.preprocessing import MinMaxScaler

# Define the columns to scale (replace with actual column names)
columns_to_scale = ['AGE', 'LEAVES USED', 'LEAVES REMAINING','PAST EXP','Days Worked','total_leaves','SALARY']

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler on the training data (assuming you have a split)
# Replace 'df' with your training data if using a split
scaler.fit(df[columns_to_scale])

# Transform the specified columns using the fitted scaler
df_2[columns_to_scale] = scaler.transform(df[columns_to_scale])


# Specify the features you want to use
features_to_use =['AGE', 'PAST EXP','designation_encoded','unit_encoded','Days Worked'] # Replace with your desired features
# Split data into training and testing sets
X = df_2[features_to_use] # Select features from the DataFrame
y = df_2['SALARY'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Gradient Boosting Regressor model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)  # Adjust n_estimators and learning_rate as needed
model.fit(X_train, y_train)

pickle.dump(model, open('Salary.pkl','wb'))

models=pickle.load(open('Salary.pkl','rb'))



