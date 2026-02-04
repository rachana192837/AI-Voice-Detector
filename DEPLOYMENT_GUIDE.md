# Deployment Guide - AI Voice Detection API

## Quick Start (Railway - Recommended)

Railway is the easiest way to deploy your API with zero configuration.

### Step 1: Install Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Or use curl
sh -c "$(curl -fsSL https://railway.app/install.sh)"
```

### Step 2: Login to Railway

```bash
railway login
```

### Step 3: Initialize Project

```bash
cd your-project-directory
railway init
```

### Step 4: Deploy

```bash
railway up
```

### Step 5: Get Your URL

```bash
railway domain
```

That's it! Your API is now live at the provided URL.

---

## Alternative Deployment Options

### Option 1: Render.com (Free Tier)

1. **Sign up at** https://render.com
2. **Create New Web Service**
3. **Connect your GitHub repository**
4. **Settings:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - Environment: Python 3
5. **Deploy!**

### Option 2: Google Cloud Run

```bash
# Install gcloud CLI
# Visit: https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud run deploy voice-detector \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

### Option 3: AWS Lambda + API Gateway

1. **Package your code:**
```bash
pip install -t package -r requirements.txt
cd package
zip -r ../deployment.zip .
cd ..
zip -g deployment.zip api.py voice_classifier_best.pth
```

2. **Create Lambda function** in AWS Console
3. **Upload deployment.zip**
4. **Add API Gateway trigger**
5. **Configure environment variables**

### Option 4: Heroku

```bash
# Install Heroku CLI
# Visit: https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main

# Scale
heroku ps:scale web=1
```

### Option 5: DigitalOcean App Platform

1. **Sign up at** https://www.digitalocean.com
2. **Create New App**
3. **Connect GitHub repository**
4. **Configure:**
   - Run Command: `uvicorn api:app --host 0.0.0.0 --port 8080`
   - HTTP Port: 8080
5. **Deploy**

---

## Docker Deployment (Any Platform)

### Build Image

```bash
docker build -t voice-detector .
```

### Test Locally

```bash
docker run -p 8000:8000 voice-detector
```

### Push to Docker Hub

```bash
# Login
docker login

# Tag
docker tag voice-detector your-username/voice-detector:latest

# Push
docker push your-username/voice-detector:latest
```

### Deploy on any cloud platform using the Docker image

---

## Environment Variables

Before deployment, set these environment variables:

```bash
export API_KEY="your-production-api-key-here"
export MODEL_PATH="voice_classifier_best.pth"
export PORT=8000
```

For Railway/Render, add these in the dashboard.

---

## Pre-Deployment Checklist

- [ ] Model trained and saved (`voice_classifier_best.pth`)
- [ ] API tested locally (`python test_api.py`)
- [ ] Unique API key generated
- [ ] All dependencies in `requirements.txt`
- [ ] Dockerfile tested
- [ ] Error handling verified
- [ ] Response format matches specification

---

## Production Configuration

### 1. Change API Key

In `api.py`, change:
```python
VALID_API_KEY = "your-secure-api-key-here"
```

To a strong, unique key:
```python
import secrets
VALID_API_KEY = secrets.token_urlsafe(32)
```

### 2. Enable Logging

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 3. Add Rate Limiting (Optional)

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/detect")
@limiter.limit("100/minute")
async def detect_voice(...):
    ...
```

### 4. Add Caching (Optional)

```bash
pip install redis
```

```python
import redis
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cache_key(audio_base64):
    return hashlib.md5(audio_base64.encode()).hexdigest()

# In detect_voice function:
cache_key = get_cache_key(request.audio_base64)
cached_result = redis_client.get(cache_key)
if cached_result:
    return json.loads(cached_result)
```

---

## Monitoring & Maintenance

### Health Checks

Your API includes:
- `/` - Basic health check
- `/health` - Detailed health check

### Logs

Check application logs regularly:

**Railway:**
```bash
railway logs
```

**Heroku:**
```bash
heroku logs --tail
```

**Render:**
Check logs in dashboard

### Performance Metrics

Monitor:
- Response time (target: < 2 seconds)
- Success rate (target: > 99%)
- Error rate
- Memory usage

---

## Troubleshooting

### Issue: "Model not found"
**Solution:** Ensure `voice_classifier_best.pth` is included in deployment

### Issue: "Out of memory"
**Solution:** 
- Reduce batch size
- Use model quantization
- Upgrade to larger instance

### Issue: "Slow response times"
**Solution:**
- Enable caching
- Use ONNX runtime
- Optimize feature extraction

### Issue: "API timing out"
**Solution:**
- Increase timeout limits in server config
- Optimize model inference
- Use async processing

---

## Cost Optimization

### Free Tier Recommendations:

1. **Railway**: Free $5 credit/month (good for ~50K requests)
2. **Render**: 750 hours/month free
3. **Google Cloud Run**: 2M requests/month free
4. **Heroku**: 550-1000 dyno hours/month free

### Tips:
- Use cold start optimization
- Implement request caching
- Use efficient model formats (ONNX)
- Monitor usage to avoid overages

---

## Security Best Practices

1. **Never commit API keys to git**
2. **Use environment variables**
3. **Enable HTTPS only**
4. **Implement rate limiting**
5. **Validate all inputs**
6. **Log all requests**
7. **Keep dependencies updated**

---

## Submission Checklist

Before submitting to the competition:

- [ ] API is publicly accessible
- [ ] Health endpoint returns 200 OK
- [ ] Detection endpoint works correctly
- [ ] API key is configured
- [ ] Response format matches specification
- [ ] Latency is < 2 seconds
- [ ] Error handling is robust
- [ ] All 5 languages are supported
- [ ] Confidence scores are calibrated
- [ ] Documentation is complete

---

## Support

If you encounter issues:

1. Check logs first
2. Test locally with `test_api.py`
3. Verify model file is present
4. Check all dependencies are installed
5. Review error messages carefully

---

## Winning Strategy Summary

To maximize your chances of winning:

1. **Accuracy** (40%):
   - Train on diverse, balanced dataset
   - Use ensemble of models
   - Fine-tune on each language

2. **Speed** (30%):
   - Optimize model inference
   - Use caching
   - Minimize feature extraction time

3. **Stability** (20%):
   - Robust error handling
   - Health monitoring
   - Comprehensive testing

4. **Explainability** (10%):
   - Return feature importance
   - Provide confidence scores
   - Add explanation in response

Good luck! 🚀
