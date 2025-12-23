# Deployment Guide - Credit Card Fraud Detection System

## Quick Start

### Option 1: Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python app.py
   ```

3. **Access the Application**
   - Web Interface: http://localhost:5000
   - API Base URL: http://localhost:5000/api

---

### Option 2: Docker Deployment

1. **Build Docker Image**
   ```bash
   docker build -t fraud-detection .
   ```

2. **Run Container**
   ```bash
   docker run -p 5000:5000 fraud-detection
   ```

3. **Using Docker Compose**
   ```bash
   docker-compose up -d
   ```

---

## Production Deployment

### Environment Variables

Create a `.env` file:
```bash
GOOGLE_API_KEY=your_google_api_key_here
FLASK_ENV=production
SECRET_KEY=your_secret_key_here
```

### Gunicorn (Recommended for Production)

1. **Install Gunicorn**
   ```bash
   pip install gunicorn
   ```

2. **Run with Gunicorn**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

### Systemd Service (Linux)

Create `/etc/systemd/system/fraud-detection.service`:
```ini
[Unit]
Description=Fraud Detection Service
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/creditcardfraud
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable fraud-detection
sudo systemctl start fraud-detection
```

---

## Cloud Deployment

### AWS EC2

1. Launch EC2 instance (Ubuntu 20.04+)
2. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv nginx
   ```
3. Clone repository and set up
4. Configure Nginx as reverse proxy
5. Use systemd service (see above)

### Heroku

1. **Create Procfile**
   ```
   web: gunicorn app:app
   ```

2. **Deploy**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Google Cloud Platform

1. Use Cloud Run:
   ```bash
   gcloud run deploy fraud-detection \
     --source . \
     --platform managed \
     --region us-central1
   ```

---

## Security Checklist

- [ ] Change default API keys
- [ ] Use HTTPS in production
- [ ] Set strong SECRET_KEY
- [ ] Configure firewall rules
- [ ] Enable rate limiting
- [ ] Set up monitoring
- [ ] Regular security updates
- [ ] Backup models and data

---

## Monitoring

### Health Check Endpoint
```bash
curl http://localhost:5000/api/model_info
```

### Logs
- Application logs: `logs/fraud_detection.log`
- Error logs: `logs/errors.log`
- Analytics: `analytics/analytics.json`

---

## Backup

Important files to backup:
- `models/*.pkl` - Trained models
- `config/api_keys.json` - API keys
- `analytics/` - Analytics data
- `.env` - Environment variables

---

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000
# Kill process
kill -9 <PID>
```

### Model Loading Errors
- Ensure models directory exists
- Check file permissions
- Verify model files are not corrupted

### API Key Issues
- Check `config/api_keys.json` exists
- Verify API key format
- Check expiration dates

