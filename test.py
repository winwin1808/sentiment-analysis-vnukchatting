import requests

# Địa chỉ API cục bộ
api_url = "http://0.0.0.0:8000/predict"

# Văn bản đầu vào
input_texts = ["Tôi rất thích sản phẩm này"]

# Tạo yêu cầu
response = requests.post(api_url, json={"texts": input_texts})

# Kiểm tra phản hồi
if response.status_code == 200:
    results = response.json()
    for result in results:
        print(f"Text: {result['text']} -> Sentiment: {result['sentiment']}")
else:
    print(f"Lỗi: {response.status_code}")