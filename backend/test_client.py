import requests
import base64
import os
import sys
import time
import psutil
import json
from typing import Dict, Any

# --- Configuration ---
SERVER_URL = "http://127.0.0.1:5000/generate"
# Use a sample image provided with TripoSR
IMAGE_PATH = os.path.join("TripoSR-main", "examples", "chair.png")
OUTPUT_PATH = "generated_model.zip"

class PerformanceMonitor:
    """Monitor CPU and memory usage during testing"""
    
    def __init__(self):
        self.start_time = None
        self.start_cpu = None
        self.start_memory = None
        
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.start_cpu = psutil.cpu_percent(interval=None)
        self.start_memory = psutil.virtual_memory().percent
        print(f"Starting test - CPU: {self.start_cpu:.1f}%, Memory: {self.start_memory:.1f}%")
        
    def end(self):
        """End monitoring and return metrics"""
        if self.start_time is None:
            return {}
            
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.virtual_memory().percent
        
        duration = end_time - self.start_time
        cpu_delta = end_cpu - self.start_cpu
        memory_delta = end_memory - self.start_memory
        
        metrics = {
            'duration': duration,
            'cpu_start': self.start_cpu,
            'cpu_end': end_cpu,
            'cpu_delta': cpu_delta,
            'memory_start': self.start_memory,
            'memory_end': end_memory,
            'memory_delta': memory_delta
        }
        
        print(f"Test completed in {duration:.2f}s")
        print(f"CPU: {self.start_cpu:.1f}% -> {end_cpu:.1f}% (Δ: {cpu_delta:+.1f}%)")
        print(f"Memory: {self.start_memory:.1f}% -> {end_memory:.1f}% (Δ: {memory_delta:+.1f}%)")
        
        return metrics

def test_generate_endpoint(params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Tests the /generate endpoint with optional parameters.
    
    Args:
        params: Optional parameters to override defaults
        
    Returns:
        Dictionary with test results and performance metrics
    """
    monitor = PerformanceMonitor()
    
    print(f"Testing with image: {IMAGE_PATH}")
    print(f"Parameters: {params or 'default'}")

    # 1. Check if the image file exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at {IMAGE_PATH}", file=sys.stderr)
        print("Please ensure you are running this script from the 'backend' directory.", file=sys.stderr)
        return {"error": "Image file not found"}

    # 2. Read and Base64-encode the image
    with open(IMAGE_PATH, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode('utf-8')
    
    image_data = f"data:image/png;base64,{encoded_string}"

    # 3. Prepare the JSON payload
    payload = {
        "sketch": image_data,
        "prompt": "a wooden chair, high quality, studio lighting",
        "preview": True,  # Use preview mode for faster testing
    }
    
    # Add custom parameters if provided
    if params:
        payload.update(params)

    print(f"Sending request to {SERVER_URL}...")
    
    # Start monitoring
    monitor.start()
    
    try:
        # 4. Send the POST request
        response = requests.post(SERVER_URL, json=payload, timeout=300)  # 5 minute timeout

        # End monitoring
        metrics = monitor.end()

        # 5. Handle the response
        if response.status_code == 200 and response.headers.get("Content-Type") == "application/zip":
            with open(OUTPUT_PATH, "wb") as f:
                f.write(response.content)
            print(f"\n✅ Success! Model saved to {os.path.abspath(OUTPUT_PATH)}")
            
            return {
                "success": True,
                "output_path": os.path.abspath(OUTPUT_PATH),
                "metrics": metrics
            }
        else:
            print(f"Error: Server returned status code {response.status_code}", file=sys.stderr)
            try:
                # Try to print the JSON error response from the server
                error_response = response.json()
                print("Server response:", error_response, file=sys.stderr)
                return {"error": error_response, "metrics": metrics}
            except requests.exceptions.JSONDecodeError:
                print("Server response:", response.text, file=sys.stderr)
                return {"error": response.text, "metrics": metrics}

    except requests.exceptions.RequestException as e:
        metrics = monitor.end()
        print(f"\nAn error occurred while sending the request: {e}", file=sys.stderr)
        print("Please ensure the backend service is running on http://127.0.0.1:5000", file=sys.stderr)
        return {"error": str(e), "metrics": metrics}

def test_health_endpoint():
    """Test the health endpoint to check server status"""
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Server Health Check:")
            print(f"  Status: {health_data.get('status')}")
            print(f"  GPU Memory: {health_data.get('gpu_mb', 0):.1f} MB")
            print(f"  CPU Usage: {health_data.get('cpu_percent', 0):.1f}%")
            print(f"  Memory Usage: {health_data.get('memory_percent', 0):.1f}%")
            print(f"  Models Loaded: {health_data.get('models_loaded', False)}")
            return health_data
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return None

def run_performance_comparison():
    """Run a comparison test between different parameter configurations"""
    print("🚀 Running Performance Comparison Test")
    print("=" * 50)
    
    # Test 1: Default optimized parameters
    print("\n📊 Test 1: Default Optimized Parameters")
    result1 = test_generate_endpoint()
    
    # Test 2: High quality parameters (more CPU intensive)
    print("\n📊 Test 2: High Quality Parameters")
    high_quality_params = {
        "num_inference_steps": 50,
        "guidance_scale": 9.0,
        "smoothing_iterations": 2,
        "mesh_threshold": 25.0,
        "preview": False
    }
    result2 = test_generate_endpoint(high_quality_params)
    
    # Test 3: Ultra fast parameters (minimal CPU usage)
    print("\n📊 Test 3: Ultra Fast Parameters")
    fast_params = {
        "num_inference_steps": 15,
        "guidance_scale": 5.0,
        "smoothing_iterations": 0,
        "mesh_threshold": 15.0,
        "preview": True
    }
    result3 = test_generate_endpoint(fast_params)
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 50)
    
    tests = [
        ("Default Optimized", result1),
        ("High Quality", result2),
        ("Ultra Fast", result3)
    ]
    
    for test_name, result in tests:
        if "metrics" in result and result["metrics"]:
            metrics = result["metrics"]
            print(f"\n{test_name}:")
            print(f"  Duration: {metrics.get('duration', 0):.2f}s")
            print(f"  CPU Delta: {metrics.get('cpu_delta', 0):+.1f}%")
            print(f"  Memory Delta: {metrics.get('memory_delta', 0):+.1f}%")
        else:
            print(f"\n{test_name}: Failed - {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    # Check server health first
    health = test_health_endpoint()
    if not health:
        print("❌ Server is not responding. Please start the backend service first.")
        sys.exit(1)
    
    # Run single test or performance comparison
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        run_performance_comparison()
    else:
        # Single test with default parameters
        result = test_generate_endpoint()
        if "error" in result:
            sys.exit(1) 