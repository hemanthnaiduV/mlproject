import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'AGE':25, 'PAST EXP':9, 'designation_encoded':6, 'unit_encoded': 3, 'Days Worked':1500})

print(r.json())

