# Deploying Joseph to Render

This guide explains how to deploy the Joseph AI text detector to Render.

## Prerequisites

1. A [Render account](https://render.com) (free tier available)
2. GitHub repository connected to Render
3. Trained model files (`joseph_v1.pkl`) - see [scripts/README.md](../scripts/README.md)

## Quick Deploy

### Option 1: Deploy from Dashboard (Recommended)

1. **Connect Repository**
   - Go to https://dashboard.render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select `joseph` repository

2. **Configure Service**
   - **Name**: `joseph` (or your preferred name)
   - **Environment**: `Docker`
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Dockerfile Path**: `./Dockerfile`

3. **Set Environment Variables**
   - Click "Advanced" → "Add Environment Variable"
   - Add these required variables:
     ```
     DATABASE_URL: (Your PostgreSQL connection string - see below)
     SECRET_KEY: (Auto-generated or use secure random string)
     OAUTH_GITHUB_CLIENT_ID: (Your GitHub OAuth app client ID)
     OAUTH_GITHUB_CLIENT_SECRET: (Your GitHub OAuth app secret)
     ```

4. **Create PostgreSQL Database** (Optional, for user authentication)
   - In Render dashboard: "New +" → "PostgreSQL"
   - Name it `joseph-db`
   - Select free tier
   - Copy the "Internal Database URL"
   - Use this as `DATABASE_URL` in your web service

5. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy automatically
   - First deploy takes ~10-15 minutes (downloading models)

### Option 2: Deploy via render.yaml (Blueprint)

1. Push `render.yaml` to your repository (already included)

2. In Render Dashboard:
   - Click "New +" → "Blueprint"
   - Select your repository
   - Render will auto-detect `render.yaml`
   - Click "Apply"

3. Set environment variables as described above

## GitHub OAuth Setup

To enable user authentication:

1. **Create GitHub OAuth App**
   - Go to GitHub Settings → Developer settings → OAuth Apps
   - Click "New OAuth App"
   - **Application name**: `Joseph AI Detector`
   - **Homepage URL**: `https://your-app-name.onrender.com`
   - **Authorization callback URL**: `https://your-app-name.onrender.com/auth/github/callback`
   - Click "Register application"

2. **Get Credentials**
   - Copy the "Client ID"
   - Generate and copy the "Client Secret"
   - Add these to Render environment variables

## Model Files

### Automatic Download (Production)

Models will be downloaded at startup if not present. Add to your startup command or use a build script:

```python
# In your app or as a script
import joblib
from pathlib import Path

model_path = Path("models/joseph_v1.pkl")
if not model_path.exists():
    # Download from your model hosting (e.g., S3, Google Cloud Storage)
    # Or train locally and upload
    pass
```

### Manual Upload

If models are small enough (<500MB), you can include them in your repository:

```bash
# Remove from .gitignore temporarily
git add models/joseph_v1.pkl
git commit -m "Add trained model for deployment"
git push
```

**Note**: Free tier has limited disk space. Consider hosting large models externally.

## Environment Variables Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `DATABASE_URL` | No* | PostgreSQL connection string | `postgresql://user:pass@host/db` |
| `SECRET_KEY` | Yes | JWT signing key | Auto-generated recommended |
| `OAUTH_GITHUB_CLIENT_ID` | No* | GitHub OAuth client ID | `Iv1.abc123...` |
| `OAUTH_GITHUB_CLIENT_SECRET` | No* | GitHub OAuth secret | `abc123def456...` |

*Required if using authentication features

## Post-Deployment

### Verify Deployment

1. Visit your app URL: `https://your-app-name.onrender.com`
2. Check health endpoint: `https://your-app-name.onrender.com/health`
3. View API docs: `https://your-app-name.onrender.com/docs`

### Monitor Logs

- In Render Dashboard → Your Service → "Logs"
- Check for startup errors
- Monitor model loading times

### Performance Tips

1. **Free Tier Limitations**:
   - App spins down after 15 min inactivity
   - First request after spindown takes ~30s
   - 512MB RAM limit

2. **Upgrade for Production**:
   - Use Starter plan ($7/month) for always-on
   - More RAM for faster model loading
   - Custom domains available

3. **Optimization**:
   - Models are cached in Docker image (faster startup)
   - Use CPU-only PyTorch (smaller image)
   - Pre-download transformers models during build

## Troubleshooting

### Build Fails

- Check Docker logs in Render dashboard
- Verify `Dockerfile` syntax
- Ensure all dependencies in `pyproject.toml`

### App Won't Start

- Check environment variables are set correctly
- Verify `DATABASE_URL` if using PostgreSQL
- Check logs for missing model files

### Slow First Request

- Normal for free tier (cold start)
- Upgrade to paid plan for always-on
- Or implement keep-alive pings

## Updating Deployment

Render automatically deploys when you push to `main`:

```bash
git add .
git commit -m "Update feature"
git push
```

Manual deploy: Render Dashboard → Your Service → "Manual Deploy" → "Deploy latest commit"

## Cost Estimate

**Free Tier**:
- Web service: Free
- PostgreSQL: Free (1GB storage)
- Total: $0/month

**Production Tier**:
- Web service (Starter): $7/month
- PostgreSQL (Starter): $7/month
- Total: $14/month

## Support

- [Render Documentation](https://render.com/docs)
- [Render Community](https://community.render.com)
- [GitHub Issues](https://github.com/JamesABaker/joseph/issues)
