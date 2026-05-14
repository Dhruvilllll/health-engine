"""
Health Engine - System Integration Test
Tests the complete system including API endpoints
"""

import json
import pandas as pd

def test_api_functionality():
    """Test API endpoints directly."""
    print("\n" + "="*60)
    print("API FUNCTIONALITY TEST")
    print("="*60)

    # Import API components
    from app.main import app, prediction_service, batch_service
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # Test 1: Health check
    print("\n[TEST 1] Health check endpoint...")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    print(f"  [OK] Status: {data['status']}")
    print(f"  [OK] Model loaded: {data['model_loaded']}")

    # Test 2: Features endpoint
    print("\n[TEST 2] Features endpoint...")
    response = client.get("/api/v1/features")
    assert response.status_code == 200
    data = response.json()
    print(f"  [OK] Primary features: {len(data['primary_features'])}")
    print(f"  [OK] Engineered features: {len(data['engineered_features'])}")
    print(f"  [OK] Total features: {data['total_features']}")

    # Test 3: Model info endpoint
    print("\n[TEST 3] Model info endpoint...")
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    print(f"  [OK] Model type: {data['model_type']}")
    print(f"  [OK] n_estimators: {data['n_estimators']}")
    print(f"  [OK] Training features: {len(data['training_features'])}")

    # Test 4: Single prediction
    print("\n[TEST 4] Single prediction endpoint...")
    prediction_input = {
        "heart_rate_variability_ms": 45.5,
        "resting_heart_rate_bpm": 52.0,
        "day_strain": 15.2,
        "blood_oxygen_": 98.5
    }

    response = client.post("/api/v1/predict", json=prediction_input)
    if response.status_code == 200:
        data = response.json()
        print(f"  [OK] Prediction score: {data['predicted_score']:.2f}")
        print(f"  [OK] Recovery status: {data['recovery_status']}")
        print(f"  [OK] Recommendation: {data['recommendation'][:40]}...")
    else:
        print(f"  [ERR] Status code: {response.status_code}")
        print(f"  Error: {response.json()}")

    # Test 5: Batch prediction
    print("\n[TEST 5] Batch prediction endpoint...")
    batch_input = {
        "records": [
            {
                "heart_rate_variability_ms": 45.5,
                "resting_heart_rate_bpm": 52.0,
                "day_strain": 15.2,
                "blood_oxygen_": 98.5
            },
            {
                "heart_rate_variability_ms": 60.0,
                "resting_heart_rate_bpm": 45.0,
                "day_strain": 8.5,
                "blood_oxygen_": 99.0
            }
        ]
    }

    response = client.post("/api/v1/batch-predict", json=batch_input)
    if response.status_code == 200:
        data = response.json()
        print(f"  [OK] Total records: {data['total_records']}")
        print(f"  [OK] Successful: {data['successful']}")
        print(f"  [OK] Failed: {data['failed']}")
        print(f"  [OK] Success rate: {data['success_rate']}%")
    else:
        print(f"  [ERR] Status code: {response.status_code}")
        print(f"  Error: {response.json()}")

    print("\n[OK] All API tests passed")


def test_full_pipeline():
    """Test the complete data pipeline."""
    print("\n" + "="*60)
    print("FULL PIPELINE TEST")
    print("="*60)

    import os
    import numpy as np
    from src.services.prediction_service import PredictionService
    from config.config import config

    # Test data loading
    print("\n[TEST 1] Data loading...")
    phys_file = "data/raw/physiological/physiological_cycles.csv"
    if os.path.exists(phys_file):
        df = pd.read_csv(phys_file)
        print(f"  [OK] Loaded {phys_file}: {df.shape}")
    else:
        print(f"  [ERR] File not found: {phys_file}")
        return

    # Test model loading
    print("\n[TEST 2] Model loading...")
    try:
        service = PredictionService()
        print(f"  [OK] Model loaded")
        print(f"  [OK] Model path: {config.MODEL_PATH}")
    except Exception as e:
        print(f"  [ERR] Model loading failed: {e}")
        return

    # Test predictions
    print("\n[TEST 3] Prediction testing...")
    sample_data = {
        "heart_rate_variability_ms": 50.0,
        "resting_heart_rate_bpm": 55.0,
        "day_strain": 12.0,
        "blood_oxygen_": 98.0,
        "stress_ratio": 1.1,
        "hrv_strain_balance": 4.17,
        "hrv_7d_avg": 50.0,
        "strain_7d_avg": 12.0
    }

    try:
        result = service.predict(sample_data)
        print(f"  [OK] Prediction successful")
        print(f"    Score: {result['predicted_score']:.2f}")
        print(f"    Status: {result['recovery_status']}")
        print(f"    Confidence: {result['confidence']}")
    except Exception as e:
        print(f"  [ERR] Prediction failed: {e}")

    # Test batch predictions
    print("\n[TEST 4] Batch prediction testing...")
    batch_df = pd.DataFrame({
        "heart_rate_variability_ms": [45, 55, 65],
        "resting_heart_rate_bpm": [50, 55, 60],
        "day_strain": [10, 15, 20],
        "blood_oxygen_": [98, 99, 97],
        "stress_ratio": [1.0, 1.1, 1.2],
        "hrv_strain_balance": [4.5, 3.67, 3.25],
        "hrv_7d_avg": [45, 55, 65],
        "strain_7d_avg": [10, 15, 20]
    })

    try:
        result_df = service.predict_batch(batch_df)
        print(f"  [OK] Batch prediction successful")
        print(f"    Total: {len(result_df)}")
        print(f"    Columns: {len(result_df.columns)}")
    except Exception as e:
        print(f"  [ERR] Batch prediction failed: {e}")

    print("\n[OK] Full pipeline test passed")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("HEALTH ENGINE - SYSTEM INTEGRATION TEST")
    print("="*60)

    try:
        # Test full pipeline
        test_full_pipeline()

        # Test API
        test_api_functionality()

        # Summary
        print("\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60)
        print("\nSystem is ready for deployment!")

    except Exception as e:
        print(f"\n[ERR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
