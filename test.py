# check_models.py - run this first
import requests, os
from dotenv import load_dotenv
load_dotenv()

r = requests.get(
    "https://integrate.api.nvidia.com/v1/models",
    headers={"Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"}
)
data = r.json().get("data", [])

print(f"Total models: {len(data)}\n")
print("=== VISION MODELS ===")
vision_keywords = ["vision", "llava", "vl", "visual", "phi", "pixtral", "molmo", "minicpm", "intern"]
for m in data:
    if any(k in m["id"].lower() for k in vision_keywords):
        print(" ", m["id"])

print("\n=== ALL MODELS ===")
for m in data:
    print(" ", m["id"])