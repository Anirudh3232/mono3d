import requests
import base64
from PIL import Image, ImageDraw
import io
import os


def test_server():
    print("\nTesting server connection...")
    try:
        response = requests.get("http://localhost:5000/test")
        print(f"Test endpoint response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to test endpoint. Make sure the server is running.")
        return False
    return True


# Create a simple test image if it doesn't exist
test_image_path = "test_sketch.png"
if not os.path.exists(test_image_path):
    print("Creating test sketch image...")
    # Create a 256x256 white image
    img = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(img)
    # Draw a simple shape (a circle)
    draw.ellipse([50, 50, 200, 200], outline='black', width=2)
    # Save the image
    img.save(test_image_path)
    print(f"Created test image at {test_image_path}")

# Read and encode the image
print("\nReading and encoding test image...")
with open(test_image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()
print("Image encoded successfully")

# Prepare the request
url = "http://localhost:5000/generate"
data = {
    "sketch": f"data:image/png;base64,{encoded_string}",
    "prompt": "a clean 3-D asset",
    "preview": False
}

print("\nSending request to service...")
print(f"URL: {url}")
print(f"Method: POST")
print(f"Data keys: {list(data.keys())}")

# First test the server connection
if not test_server():
    exit(1)

# Send the request
try:
    response = requests.post(url, json=data)
    print(f"\nResponse status code: {response.status_code}")
    if response.status_code == 200:
        print("✅ Service is working!")
        print("Response received successfully")
    else:
        print(f"❌ Error: {response.status_code}")
        print("Response text:", response.text)
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the service. Make sure it's running on port 5000")
except Exception as e:
    print(f"❌ Unexpected error: {str(e)}")
