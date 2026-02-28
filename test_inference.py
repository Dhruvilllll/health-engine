from inference import HealthInferenceEngine

engine = HealthInferenceEngine()

# Test cases
test_cases = [
    (65, 98),     # Rest condition
    (120, 97),    # Moderate effort
    (160, 95),    # High effort
    (185, 91)     # Critical condition
]

for hr, spo2 in test_cases:
    result = engine.predict(hr, spo2)
    print(f"\nHR={hr}, SpO2={spo2}")
    print(result)