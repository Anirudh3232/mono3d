#!/usr/bin/env python3
"""
Mono3D Optimization Test Script

This script tests the optimization improvements and demonstrates
the performance gains achieved through the CPU optimization work.
"""

import requests
import time
import psutil
import json
import base64
import os
from typing import Dict, Any

# Configuration
SERVER_URL = "http://127.0.0.1:5000"
IMAGE_PATH = os.path.join("TripoSR-main", "examples", "chair.png")

class OptimizationTester:
    """Test the optimization improvements"""
    
    def __init__(self):
        self.results = {}
        
    def check_server_health(self) -> bool:
        """Check if the server is running and healthy"""
        try:
            response = requests.get(f"{SERVER_URL}/health", timeout=10)
            if response.status_code == 200:
                health = response.json()
                print("✅ Server Health Check:")
                print(f"  Status: {health.get('status')}")
                print(f"  GPU Memory: {health.get('gpu_mb', 0):.1f} MB")
                print(f"  CPU Usage: {health.get('cpu_percent', 0):.1f}%")
                print(f"  Memory Usage: {health.get('memory_percent', 0):.1f}%")
                print(f"  Models Loaded: {health.get('models_loaded', False)}")
                print(f"  Optimization Available: {health.get('optimization_available', False)}")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    def get_available_profiles(self) -> Dict[str, str]:
        """Get available optimization profiles"""
        try:
            response = requests.get(f"{SERVER_URL}/profiles", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("profiles", {})
            else:
                print(f"❌ Failed to get profiles: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ Error getting profiles: {e}")
            return {}
    
    def get_system_recommendation(self, use_case: str = "general") -> str:
        """Get system recommendation"""
        try:
            response = requests.get(f"{SERVER_URL}/recommend?use_case={use_case}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                recommended = data.get("recommended_profile")
                specs = data.get("system_specs", {})
                print(f"🎯 System Recommendation for {use_case}:")
                print(f"  Recommended Profile: {recommended}")
                print(f"  CPU Cores: {specs.get('cpu_cores', 0)}")
                print(f"  GPU Memory: {specs.get('gpu_memory_gb', 0):.1f} GB")
                return recommended
            else:
                print(f"❌ Failed to get recommendation: {response.status_code}")
                return "standard"
        except Exception as e:
            print(f"❌ Error getting recommendation: {e}")
            return "standard"
    
    def load_test_image(self) -> str:
        """Load and encode test image"""
        if not os.path.exists(IMAGE_PATH):
            print(f"❌ Test image not found: {IMAGE_PATH}")
            return None
        
        with open(IMAGE_PATH, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"
    
    def test_generation(self, profile: str = None, custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test generation with specific profile"""
        image_data = self.load_test_image()
        if not image_data:
            return {"error": "No test image available"}
        
        payload = {
            "sketch": image_data,
            "prompt": "a wooden chair, high quality, studio lighting",
            "preview": True
        }
        
        if profile:
            payload["profile"] = profile
        
        if custom_params:
            payload["custom_params"] = custom_params
        
        print(f"\n🚀 Testing generation with profile: {profile or 'default'}")
        
        # Monitor system resources
        cpu_start = psutil.cpu_percent(interval=None)
        memory_start = psutil.virtual_memory().percent
        start_time = time.time()
        
        try:
            response = requests.post(f"{SERVER_URL}/generate", json=payload, timeout=300)
            
            end_time = time.time()
            cpu_end = psutil.cpu_percent(interval=None)
            memory_end = psutil.virtual_memory().percent
            
            duration = end_time - start_time
            cpu_delta = cpu_end - cpu_start
            memory_delta = memory_end - memory_start
            
            result = {
                "success": response.status_code == 200,
                "duration": duration,
                "cpu_start": cpu_start,
                "cpu_end": cpu_end,
                "cpu_delta": cpu_delta,
                "memory_start": memory_start,
                "memory_end": memory_end,
                "memory_delta": memory_delta,
                "profile": profile
            }
            
            if response.status_code == 200:
                print(f"✅ Generation successful in {duration:.2f}s")
                print(f"  CPU: {cpu_start:.1f}% -> {cpu_end:.1f}% (Δ: {cpu_delta:+.1f}%)")
                print(f"  Memory: {memory_start:.1f}% -> {memory_end:.1f}% (Δ: {memory_delta:+.1f}%)")
            else:
                print(f"❌ Generation failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"  Error: {error_data}")
                except:
                    print(f"  Error: {response.text}")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"❌ Generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "profile": profile
            }
    
    def run_optimization_comparison(self):
        """Run comparison test between different profiles"""
        print("🔬 Running Optimization Comparison Test")
        print("=" * 60)
        
        # Test different profiles
        profiles_to_test = [
            ("ultra_fast", "Ultra Fast (Minimal CPU)"),
            ("fast", "Fast (Balanced)"),
            ("standard", "Standard (Default)"),
            ("high_quality", "High Quality"),
            ("maximum_quality", "Maximum Quality")
        ]
        
        results = {}
        
        for profile_name, description in profiles_to_test:
            print(f"\n📊 Testing {description}")
            print("-" * 40)
            
            result = self.test_generation(profile_name)
            results[profile_name] = result
            
            # Wait between tests to let system stabilize
            time.sleep(5)
        
        # Summary
        print("\n" + "=" * 60)
        print("📈 OPTIMIZATION COMPARISON SUMMARY")
        print("=" * 60)
        
        successful_results = {k: v for k, v in results.items() if v.get("success", False)}
        
        if not successful_results:
            print("❌ No successful generations to compare")
            return
        
        # Sort by duration
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]["duration"])
        
        print(f"\n🏆 Performance Ranking (by speed):")
        for i, (profile, result) in enumerate(sorted_results, 1):
            print(f"{i}. {profile}: {result['duration']:.2f}s (CPU Δ: {result['cpu_delta']:+.1f}%)")
        
        print(f"\n📊 Detailed Results:")
        for profile, result in sorted_results:
            print(f"\n{profile.upper()}:")
            print(f"  Duration: {result['duration']:.2f}s")
            print(f"  CPU Usage: {result['cpu_start']:.1f}% -> {result['cpu_end']:.1f}% (Δ: {result['cpu_delta']:+.1f}%)")
            print(f"  Memory Usage: {result['memory_start']:.1f}% -> {result['memory_end']:.1f}% (Δ: {result['memory_delta']:+.1f}%)")
        
        # Calculate improvements
        if len(sorted_results) >= 2:
            fastest = sorted_results[0]
            slowest = sorted_results[-1]
            
            speed_improvement = (slowest[1]["duration"] - fastest[1]["duration"]) / slowest[1]["duration"] * 100
            cpu_improvement = fastest[1]["cpu_delta"] - slowest[1]["cpu_delta"]
            
            print(f"\n🎯 Key Improvements:")
            print(f"  Speed: {fastest[0]} is {speed_improvement:.1f}% faster than {slowest[0]}")
            print(f"  CPU: {fastest[0]} uses {cpu_improvement:.1f}% less CPU than {slowest[0]}")
    
    def test_caching(self):
        """Test the caching system"""
        print("\n💾 Testing Caching System")
        print("=" * 40)
        
        # First request (should be slow)
        print("📤 First request (cache miss)...")
        result1 = self.test_generation("fast")
        
        time.sleep(2)
        
        # Second request (should be fast due to caching)
        print("📤 Second request (cache hit)...")
        result2 = self.test_generation("fast")
        
        if result1.get("success") and result2.get("success"):
            speedup = result1["duration"] / result2["duration"]
            print(f"\n🎯 Caching Performance:")
            print(f"  First request: {result1['duration']:.2f}s")
            print(f"  Second request: {result2['duration']:.2f}s")
            print(f"  Speedup: {speedup:.1f}x faster")
            
            if speedup > 2:
                print("  ✅ Caching is working effectively!")
            else:
                print("  ⚠️  Caching may not be working as expected")
        else:
            print("❌ Cannot test caching - generation failed")

def main():
    """Main test function"""
    print("🚀 Mono3D Optimization Test Suite")
    print("=" * 50)
    
    tester = OptimizationTester()
    
    # Check server health
    if not tester.check_server_health():
        print("❌ Server is not available. Please start the backend service first.")
        return
    
    # Get available profiles
    profiles = tester.get_available_profiles()
    if profiles:
        print(f"\n📋 Available Optimization Profiles:")
        for name, description in profiles.items():
            print(f"  {name}: {description}")
    
    # Get system recommendation
    recommended = tester.get_system_recommendation()
    
    # Test caching
    tester.test_caching()
    
    # Run optimization comparison
    tester.run_optimization_comparison()
    
    print("\n✅ Optimization testing completed!")
    print(f"💡 Recommended profile for your system: {recommended}")

if __name__ == "__main__":
    main() 