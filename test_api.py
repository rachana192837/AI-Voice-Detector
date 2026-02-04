"""
API Testing Script
Test your voice detection API locally before submission
"""

import requests
import base64
import json
import time
from pathlib import Path


class APITester:
    """Test the voice detection API"""
    
    def __init__(self, base_url="http://localhost:8000", api_key="your-secure-api-key-here"):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def test_health(self):
        """Test health endpoint"""
        print("\n" + "="*60)
        print("Testing Health Endpoint")
        print("="*60)
        
        try:
            response = requests.get(f"{self.base_url}/health")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                print("✓ Health check passed!")
                return True
            else:
                print("✗ Health check failed!")
                return False
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return False
    
    def encode_audio_file(self, audio_path):
        """Encode audio file to base64"""
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return audio_base64
    
    def test_detection(self, audio_path, language="english"):
        """Test voice detection endpoint"""
        print("\n" + "="*60)
        print(f"Testing Detection: {audio_path}")
        print("="*60)
        
        try:
            # Encode audio
            print("Encoding audio...")
            audio_base64 = self.encode_audio_file(audio_path)
            
            # Prepare request
            payload = {
                "audio_base64": audio_base64,
                "language": language
            }
            
            # Send request
            print("Sending request...")
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/detect",
                json=payload,
                headers=self.headers
            )
            response_time = (time.time() - start_time) * 1000
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Time: {response_time:.2f} ms")
            
            if response.status_code == 200:
                result = response.json()
                print("\n📊 Detection Result:")
                print(f"  Classification: {result['classification']}")
                print(f"  Confidence: {result['confidence']:.4f}")
                print(f"  Language: {result['language']}")
                print(f"  Processing Time: {result['processing_time_ms']:.2f} ms")
                
                if 'explanation' in result:
                    print(f"\n📝 Explanation:")
                    for key, value in result['explanation'].items():
                        print(f"  {key}: {value}")
                
                print("\n✓ Detection test passed!")
                return True, result
            else:
                print(f"✗ Error: {response.text}")
                return False, None
                
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return False, None
    
    def test_invalid_api_key(self):
        """Test with invalid API key"""
        print("\n" + "="*60)
        print("Testing Invalid API Key")
        print("="*60)
        
        try:
            invalid_headers = {
                "X-API-Key": "invalid-key",
                "Content-Type": "application/json"
            }
            
            payload = {
                "audio_base64": "dummy",
                "language": "english"
            }
            
            response = requests.post(
                f"{self.base_url}/detect",
                json=payload,
                headers=invalid_headers
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 401:
                print("✓ Correctly rejected invalid API key!")
                return True
            else:
                print("✗ Should have rejected invalid API key!")
                return False
                
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return False
    
    def test_invalid_language(self, audio_path):
        """Test with invalid language"""
        print("\n" + "="*60)
        print("Testing Invalid Language")
        print("="*60)
        
        try:
            audio_base64 = self.encode_audio_file(audio_path)
            
            payload = {
                "audio_base64": audio_base64,
                "language": "spanish"  # Not supported
            }
            
            response = requests.post(
                f"{self.base_url}/detect",
                json=payload,
                headers=self.headers
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 422:  # Validation error
                print("✓ Correctly rejected invalid language!")
                return True
            else:
                print("✗ Should have rejected invalid language!")
                return False
                
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return False
    
    def test_multiple_requests(self, audio_path, num_requests=10):
        """Test multiple requests for stability"""
        print("\n" + "="*60)
        print(f"Testing Multiple Requests ({num_requests})")
        print("="*60)
        
        audio_base64 = self.encode_audio_file(audio_path)
        
        successes = 0
        failures = 0
        response_times = []
        
        for i in range(num_requests):
            payload = {
                "audio_base64": audio_base64,
                "language": "english"
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/detect",
                json=payload,
                headers=self.headers
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                successes += 1
                response_times.append(response_time)
            else:
                failures += 1
            
            print(f"Request {i+1}/{num_requests}: {'✓' if response.status_code == 200 else '✗'} ({response_time:.2f} ms)")
        
        print(f"\n📊 Results:")
        print(f"  Successes: {successes}/{num_requests}")
        print(f"  Failures: {failures}/{num_requests}")
        
        if response_times:
            print(f"  Avg Response Time: {sum(response_times)/len(response_times):.2f} ms")
            print(f"  Min Response Time: {min(response_times):.2f} ms")
            print(f"  Max Response Time: {max(response_times):.2f} ms")
        
        if successes == num_requests:
            print("\n✓ All requests succeeded!")
            return True
        else:
            print(f"\n✗ {failures} requests failed!")
            return False
    
    def run_all_tests(self, test_audio_path):
        """Run all tests"""
        print("\n" + "="*60)
        print("🧪 Running Complete API Test Suite")
        print("="*60)
        
        results = {}
        
        # Test 1: Health check
        results['health'] = self.test_health()
        
        # Test 2: Valid detection
        results['detection'], _ = self.test_detection(test_audio_path)
        
        # Test 3: Invalid API key
        results['invalid_key'] = self.test_invalid_api_key()
        
        # Test 4: Invalid language
        results['invalid_language'] = self.test_invalid_language(test_audio_path)
        
        # Test 5: Multiple requests
        results['stability'] = self.test_multiple_requests(test_audio_path, num_requests=5)
        
        # Summary
        print("\n" + "="*60)
        print("📊 Test Summary")
        print("="*60)
        
        total_tests = len(results)
        passed_tests = sum(1 for v in results.values() if v)
        
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{test_name.ljust(20)}: {status}")
        
        print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 All tests passed! API is ready for submission!")
        else:
            print("\n⚠️  Some tests failed. Please fix the issues before submission.")
        
        return passed_tests == total_tests


def main():
    """Main function"""
    print("="*60)
    print("AI Voice Detection API - Testing Tool")
    print("="*60)
    
    # Configuration
    BASE_URL = input("\nEnter API URL (default: http://localhost:8000): ").strip() or "http://localhost:8000"
    API_KEY = input("Enter API Key (default: your-secure-api-key-here): ").strip() or "your-secure-api-key-here"
    
    # Get test audio file
    print("\nEnter path to test audio file:")
    print("(Place a sample MP3 file in the current directory)")
    test_audio = input("Audio file path (default: test_sample.mp3): ").strip() or "test_sample.mp3"
    
    if not Path(test_audio).exists():
        print(f"\n✗ Error: Audio file '{test_audio}' not found!")
        print("Please provide a valid audio file path.")
        return
    
    # Initialize tester
    tester = APITester(base_url=BASE_URL, api_key=API_KEY)
    
    # Run tests
    success = tester.run_all_tests(test_audio)
    
    if success:
        print("\n" + "="*60)
        print("🚀 Next Steps:")
        print("="*60)
        print("1. Deploy your API to a public server")
        print("2. Update the API URL in your submission")
        print("3. Submit your endpoint and API key")
        print("4. Monitor your API during evaluation")
        print("="*60)
    

if __name__ == "__main__":
    main()
