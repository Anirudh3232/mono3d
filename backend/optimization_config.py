"""
Optimization Configuration for Mono3D Service

This module defines different optimization profiles to balance quality vs performance.
Each profile is designed for specific use cases and CPU/GPU constraints.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class OptimizationProfile:
    """Defines a set of optimization parameters for the 3D image generation pipeline"""
    
    name: str
    description: str
    
    # Stable Diffusion parameters
    num_inference_steps: int
    guidance_scale: float
    
    # 3D rendering parameters
    render_resolution: int
    
    # Memory optimization
    enable_caching: bool
    cache_size: int
    
    # Performance hints
    estimated_duration: str
    cpu_usage: str
    quality_level: str

class OptimizationProfiles:
    """Predefined optimization profiles for different use cases"""
    
    # Ultra Fast - Minimal CPU usage, basic quality
    ULTRA_FAST = OptimizationProfile(
        name="ultra_fast",
        description="Minimal CPU usage for rapid prototyping",
        num_inference_steps=15,
        guidance_scale=5.0,
        render_resolution=256,
        enable_caching=True,
        cache_size=20,
        estimated_duration="30-60 seconds",
        cpu_usage="Low",
        quality_level="Basic"
    )
    
    # Fast - Balanced performance and quality
    FAST = OptimizationProfile(
        name="fast",
        description="Balanced performance for general use",
        num_inference_steps=25,
        guidance_scale=6.5,
        render_resolution=384,
        enable_caching=True,
        cache_size=15,
        estimated_duration="60-120 seconds",
        cpu_usage="Medium",
        quality_level="Good"
    )
    
    # Standard - Default optimized settings
    STANDARD = OptimizationProfile(
        name="standard",
        description="Optimized default settings",
        num_inference_steps=30,
        guidance_scale=7.5,
        render_resolution=512,
        enable_caching=True,
        cache_size=10,
        estimated_duration="90-180 seconds",
        cpu_usage="Medium-High",
        quality_level="Very Good"
    )
    
    # High Quality - Better quality, higher CPU usage
    HIGH_QUALITY = OptimizationProfile(
        name="high_quality",
        description="High quality output with moderate CPU usage",
        num_inference_steps=40,
        guidance_scale=8.5,
        render_resolution=768,
        enable_caching=True,
        cache_size=8,
        estimated_duration="120-240 seconds",
        cpu_usage="High",
        quality_level="Excellent"
    )
    
    # Maximum Quality - Best quality, highest CPU usage
    MAXIMUM_QUALITY = OptimizationProfile(
        name="maximum_quality",
        description="Maximum quality output (CPU intensive)",
        num_inference_steps=50,
        guidance_scale=9.5,
        render_resolution=1024,
        enable_caching=False,  # Disable caching for maximum quality
        cache_size=0,
        estimated_duration="180-300 seconds",
        cpu_usage="Very High",
        quality_level="Maximum"
    )

def get_profile(profile_name: str) -> OptimizationProfile:
    """Get an optimization profile by name"""
    profiles = {
        "ultra_fast": OptimizationProfiles.ULTRA_FAST,
        "fast": OptimizationProfiles.FAST,
        "standard": OptimizationProfiles.STANDARD,
        "high_quality": OptimizationProfiles.HIGH_QUALITY,
        "maximum_quality": OptimizationProfiles.MAXIMUM_QUALITY,
    }
    
    if profile_name not in profiles:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(profiles.keys())}")
    
    return profiles[profile_name]

def get_profile_parameters(profile_name: str, custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get parameters for a specific profile with optional custom overrides"""
    profile = get_profile(profile_name)
    
    params = {
        'num_inference_steps': profile.num_inference_steps,
        'guidance_scale': profile.guidance_scale,
        'render_resolution': profile.render_resolution,
        'enable_caching': profile.enable_caching,
        'cache_size': profile.cache_size,
    }
    
    # Apply custom overrides if provided
    if custom_params:
        params.update(custom_params)
    
    return params

def list_profiles() -> Dict[str, str]:
    """List all available profiles with descriptions"""
    profiles = {
        "ultra_fast": "Minimal CPU usage for rapid prototyping",
        "fast": "Balanced performance for general use", 
        "standard": "Optimized default settings",
        "high_quality": "High quality output with moderate CPU usage",
        "maximum_quality": "Maximum quality output (CPU intensive)"
    }
    return profiles

def get_recommended_profile(cpu_cores: int, gpu_memory_gb: float, use_case: str = "general") -> str:
    """Get recommended profile based on system specs and use case"""
    
    if use_case == "prototyping":
        return "ultra_fast"
    elif use_case == "production":
        if cpu_cores >= 8 and gpu_memory_gb >= 8:
            return "high_quality"
        elif cpu_cores >= 4 and gpu_memory_gb >= 4:
            return "standard"
        else:
            return "fast"
    elif use_case == "maximum_quality":
        if cpu_cores >= 12 and gpu_memory_gb >= 12:
            return "maximum_quality"
        else:
            return "high_quality"
    else:  # general use case
        if cpu_cores >= 6 and gpu_memory_gb >= 6:
            return "standard"
        else:
            return "fast"

# Example usage and testing
if __name__ == "__main__":
    print("Available Optimization Profiles:")
    print("=" * 50)
    
    for name, description in list_profiles().items():
        profile = get_profile(name)
        print(f"\n{name.upper()}:")
        print(f"  Description: {description}")
        print(f"  Inference Steps: {profile.num_inference_steps}")
        print(f"  Guidance Scale: {profile.guidance_scale}")
        print(f"  Resolution: {profile.render_resolution}")
        print(f"  Estimated Duration: {profile.estimated_duration}")
        print(f"  CPU Usage: {profile.cpu_usage}")
        print(f"  Quality Level: {profile.quality_level}")
    
    print("\n" + "=" * 50)
    print("Example: Get parameters for 'standard' profile")
    params = get_profile_parameters("standard")
    print(f"Parameters: {params}")
    
    print("\nExample: Get recommended profile for system with 8 CPU cores and 8GB GPU")
    recommended = get_recommended_profile(8, 8.0, "production")
    print(f"Recommended: {recommended}") 