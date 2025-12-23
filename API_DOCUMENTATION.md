# API Documentation - Credit Card Fraud Detection System

## Base URL
```
http://localhost:5000
```

## Authentication
All API endpoints (except `/api/generate_key` and `/api/validate_key`) require API key authentication.

### Methods to provide API key:
1. **Header**: `X-API-Key: your_api_key_here`
2. **Header**: `Authorization: Bearer your_api_key_here`
3. **Query Parameter**: `?api_key=your_api_key_here`
4. **JSON Body**: `{"api_key": "your_api_key_here"}`

---

## Endpoints

### 1. Generate API Key
**POST** `/api/generate_key`

Generate a new API key for authentication.

**Request Body:**
```json
{
  "name": "My API Key",
  "expires_days": 30
}
```

**Response:**
```json
{
  "success": true,
  "api_key": "fraud_api_...",
  "message": "API key generated successfully. Save it securely!"
}
```

---

### 2. Validate API Key
**POST** `/api/validate_key`

Validate an API key.

**Request Body:**
```json
{
  "api_key": "your_api_key_here"
}
```

**Response:**
```json
{
  "valid": true,
  "key_info": {
    "name": "My API Key",
    "usage_count": 42,
    "last_used": "2024-01-15T10:30:00"
  }
}
```

---

### 3. Predict Fraud (Single Transaction)
**POST** `/api/predict`

Predict fraud for a single transaction.

**Headers:**
- `X-API-Key: your_api_key`
- `Content-Type: application/json`

**Request Body:**
```json
{
  "Time": 12345,
  "V1": -1.23,
  "V2": 0.45,
  "V3": 0.12,
  ...
  "V28": 0.34,
  "Amount": 100.50
}
```

**Response:**
```json
{
  "prediction": 0,
  "is_fraud": false,
  "fraud_probability": 0.0234,
  "not_fraud_probability": 0.9766,
  "confidence": 0.9766,
  "model_used": "Ensemble"
}
```

---

### 4. Batch Predictions
**POST** `/api/batch_predict`

Predict fraud for multiple transactions at once.

**Headers:**
- `X-API-Key: your_api_key`
- `Content-Type: application/json`

**Request Body:**
```json
{
  "transactions": [
    {
      "Time": 12345,
      "V1": -1.23,
      "V2": 0.45,
      ...
      "Amount": 100.50
    },
    {
      "Time": 12350,
      "V1": 2.34,
      "V2": -0.56,
      ...
      "Amount": 250.00
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "total": 2,
  "results": [
    {
      "index": 0,
      "prediction": 0,
      "is_fraud": false,
      "fraud_probability": 0.0234,
      "not_fraud_probability": 0.9766,
      "confidence": 0.9766
    },
    {
      "index": 1,
      "prediction": 1,
      "is_fraud": true,
      "fraud_probability": 0.8765,
      "not_fraud_probability": 0.1235,
      "confidence": 0.8765
    }
  ]
}
```

---

### 5. Get Analytics
**GET** `/api/analytics`

Get real-time analytics and dashboard statistics.

**Headers:**
- `X-API-Key: your_api_key`

**Response:**
```json
{
  "success": true,
  "analytics": {
    "total_predictions": 1000,
    "fraud_detected": 17,
    "safe_transactions": 983,
    "fraud_rate": 1.7,
    "total_amount_checked": 125000.50,
    "fraud_amount_detected": 2500.00,
    "daily_trend": [...]
  }
}
```

---

### 6. Train Model
**POST** `/api/train`

Train a new fraud detection model.

**Headers:**
- `X-API-Key: your_api_key`

**Response:**
```json
{
  "success": true,
  "message": "Model trained successfully",
  "metrics": {
    "best_model": "XGBoost",
    "roc_auc": 0.9856,
    "avg_precision": 0.9234,
    "models": {...}
  }
}
```

---

### 7. Get Model Info
**GET** `/api/model_info`

Get information about the current model.

**Response:**
```json
{
  "loaded": true,
  "model_name": "XGBoost",
  "metrics": {
    "roc_auc": 0.9856,
    "avg_precision": 0.9234,
    "trained_at": "2024-01-15T10:00:00"
  }
}
```

---

### 8. List All Models
**GET** `/api/models`

List all available models and their performance.

**Headers:**
- `X-API-Key: your_api_key`

**Response:**
```json
{
  "success": true,
  "models": {
    "current_model": "XGBoost",
    "available_models": [
      {
        "name": "XGBoost",
        "roc_auc": 0.9856,
        "avg_precision": 0.9234,
        "is_active": true
      },
      {
        "name": "LightGBM",
        "roc_auc": 0.9823,
        "avg_precision": 0.9156,
        "is_active": false
      }
    ]
  }
}
```

---

### 9. Generate Sample Transaction
**GET** `/api/generate_sample`

Generate a random sample transaction for testing.

**Headers:**
- `X-API-Key: your_api_key`

**Response:**
```json
{
  "Time": 12345,
  "V1": -1.2345,
  "V2": 0.5678,
  ...
  "V28": 0.1234,
  "Amount": 88.50
}
```

---

### 10. Chatbot
**POST** `/api/chatbot`

Get AI-powered responses about fraud detection.

**Headers:**
- `X-API-Key: your_api_key`
- `Content-Type: application/json`

**Request Body:**
```json
{
  "message": "What features are most important for fraud detection?",
  "history": []
}
```

**Response:**
```json
{
  "success": true,
  "response": "The most important features for fraud detection include..."
}
```

---

### 11. Explain Prediction
**POST** `/api/explain_prediction`

Get AI-generated explanation of a fraud prediction.

**Headers:**
- `X-API-Key: your_api_key`
- `Content-Type: application/json`

**Request Body:**
```json
{
  "prediction": {
    "is_fraud": true,
    "fraud_probability": 0.8765,
    "not_fraud_probability": 0.1235,
    "confidence": 0.8765
  }
}
```

**Response:**
```json
{
  "success": true,
  "explanation": "This transaction was flagged as fraudulent because..."
}
```

---

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Error message here"
}
```

**HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid input)
- `401`: Unauthorized (invalid/missing API key)
- `500`: Internal Server Error

---

## Rate Limiting

Currently, there are no rate limits, but it's recommended to:
- Batch multiple predictions together when possible
- Cache results when appropriate
- Use batch prediction endpoint for multiple transactions

---

## Example Usage

### Python
```python
import requests

API_KEY = "your_api_key_here"
BASE_URL = "http://localhost:5000"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Single prediction
response = requests.post(
    f"{BASE_URL}/api/predict",
    headers=headers,
    json={
        "Time": 12345,
        "V1": -1.23,
        "V2": 0.45,
        # ... other features
        "Amount": 100.50
    }
)

result = response.json()
print(f"Fraud: {result['is_fraud']}, Probability: {result['fraud_probability']}")
```

### cURL
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 12345,
    "V1": -1.23,
    "Amount": 100.50
  }'
```

