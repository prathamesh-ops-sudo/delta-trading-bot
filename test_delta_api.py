#!/usr/bin/env python3
"""
Test Delta Exchange API endpoints to find correct format
"""
import requests
import time

# Test different API formats
base_url = "https://api.delta.exchange"

print("Testing Delta Exchange API formats...")
print("=" * 60)

# Format 1: With count
print("\n1. Testing with count parameter:")
url = f"{base_url}/v2/history/candles"
params = {
    "symbol": "BTCUSD",
    "resolution": "5m",
    "count": 10
}
print(f"URL: {url}")
print(f"Params: {params}")
response = requests.get(url, params=params)
print(f"Status: {response.status_code}")
if response.status_code != 200:
    print(f"Error: {response.text[:200]}")
else:
    print(f"Success! Got {len(response.json().get('result', []))} candles")

# Format 2: With start/end
print("\n2. Testing with start/end timestamps:")
end_time = int(time.time())
start_time = end_time - (10 * 300)  # 10 candles * 5min
params = {
    "symbol": "BTCUSD",
    "resolution": "5m",
    "start": start_time,
    "end": end_time
}
print(f"URL: {url}")
print(f"Params: {params}")
response = requests.get(url, params=params)
print(f"Status: {response.status_code}")
if response.status_code != 200:
    print(f"Error: {response.text[:200]}")
else:
    print(f"Success! Got {len(response.json().get('result', []))} candles")

# Format 3: Just symbol and resolution
print("\n3. Testing with just symbol and resolution:")
params = {
    "symbol": "BTCUSD",
    "resolution": "5m"
}
print(f"URL: {url}")
print(f"Params: {params}")
response = requests.get(url, params=params)
print(f"Status: {response.status_code}")
if response.status_code != 200:
    print(f"Error: {response.text[:200]}")
else:
    result = response.json().get('result', [])
    print(f"Success! Got {len(result)} candles")
    if result:
        print(f"Sample candle: {result[0]}")

# Format 4: Check products endpoint
print("\n4. Testing products endpoint:")
url = f"{base_url}/v2/products/BTCUSD"
response = requests.get(url)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    product = response.json().get('result', {})
    print(f"Product symbol: {product.get('symbol')}")
    print(f"Product ID: {product.get('id')}")

print("\n" + "=" * 60)
print("Test complete!")
