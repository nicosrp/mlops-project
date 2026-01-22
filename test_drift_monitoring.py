"""
Test script to verify data drift monitoring is working.
"""

import time
from pathlib import Path

import numpy as np
import requests


def test_api_health():
    """Test if API is running."""
    print("1. Testing API health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("   ✓ API is running")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"   ✗ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ✗ API is not running. Start it with: uvicorn mlops_project.api:app --reload")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def generate_test_predictions(n=30):
    """Generate test predictions."""
    print(f"\n2. Generating {n} test predictions...")
    
    successful = 0
    failed = 0
    
    for i in range(n):
        try:
            # Generate random features: 5 historical matches, 22 features each
            features = np.random.randn(5, 22).tolist()
            
            response = requests.post(
                "http://localhost:8000/predict",
                json={"features": features},
                timeout=5
            )
            
            if response.status_code == 200:
                successful += 1
                result = response.json()
                if (i + 1) % 10 == 0:
                    print(f"   Prediction {i+1}: {result['prediction']} (probs: {result['probabilities']})")
            else:
                failed += 1
                print(f"   ✗ Prediction {i+1} failed: {response.status_code}")
        
        except Exception as e:
            failed += 1
            print(f"   ✗ Prediction {i+1} error: {e}")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    print(f"   ✓ Completed: {successful} successful, {failed} failed")
    return successful > 0


def check_database():
    """Check if prediction database was created."""
    print("\n3. Checking prediction database...")
    db_path = Path("data/prediction_database.csv")
    
    if db_path.exists():
        import pandas as pd
        df = pd.read_csv(db_path)
        print(f"   ✓ Database exists: {db_path}")
        print(f"   ✓ Total predictions logged: {len(df)}")
        print(f"   ✓ Columns: {list(df.columns)}")
        
        if len(df) > 0:
            print(f"\n   Sample data:")
            print(df.head(3).to_string(index=False))
        
        return True
    else:
        print(f"   ✗ Database not found at {db_path}")
        return False


def test_monitoring_endpoint():
    """Test the monitoring endpoint."""
    print("\n4. Testing monitoring endpoint...")
    try:
        response = requests.get("http://localhost:8000/monitoring", timeout=30)
        
        if response.status_code == 200:
            html_content = response.text
            print("   ✓ Monitoring endpoint works")
            print(f"   ✓ Response length: {len(html_content)} characters")
            
            # Check if it's actually HTML
            if "<html" in html_content.lower():
                print("   ✓ Valid HTML response received")
                print("\n   Open in browser: http://localhost:8000/monitoring")
                return True
            else:
                print(f"   ⚠ Response: {html_content[:200]}")
                return False
        else:
            print(f"   ✗ Status code: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
    
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_standalone_script():
    """Test the standalone drift monitoring script."""
    print("\n5. Testing standalone monitoring script...")
    try:
        from src.mlops_project.data_drift import load_current_data, load_reference_data
        
        current = load_current_data()
        print(f"   ✓ Loaded current data: {current.shape}")
        
        if len(current) >= 20:
            print("   ✓ Sufficient data for drift analysis")
            print("\n   You can run: python src/mlops_project/data_drift.py")
            return True
        else:
            print(f"   ⚠ Only {len(current)} predictions. Need at least 20 for full analysis.")
            return False
    
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DATA DRIFT MONITORING TEST")
    print("=" * 60)
    
    results = []
    
    # Test 1: API Health
    results.append(("API Health", test_api_health()))
    
    if not results[0][1]:
        print("\n⚠ API is not running. Please start it first:")
        print("   cd src")
        print("   python -m mlops_project.api")
        print("\nor:")
        print("   uvicorn mlops_project.api:app --reload")
        return
    
    # Test 2: Generate predictions
    results.append(("Generate Predictions", generate_test_predictions(30)))
    
    # Test 3: Check database
    results.append(("Database Created", check_database()))
    
    # Test 4: Monitoring endpoint
    results.append(("Monitoring Endpoint", test_monitoring_endpoint()))
    
    # Test 5: Standalone script
    results.append(("Standalone Script", test_standalone_script()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! Data drift monitoring is working correctly.")
        print("\nNext steps:")
        print("1. View monitoring dashboard: http://localhost:8000/monitoring")
        print("2. Run standalone analysis: python src/mlops_project/data_drift.py")
        print("3. Check the report: reports/data_drift_report.html")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
