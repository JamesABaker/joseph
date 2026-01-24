# User Account Setup Guide

This guide explains how to set up OAuth authentication with GitHub for local testing and development.

## Prerequisites

- Docker and Docker Compose installed
- GitHub account (for GitHub OAuth)

## Setup Steps

### 1. Configure OAuth Providers

#### GitHub OAuth Setup

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click "New OAuth App"
3. Fill in application details:
   - Application name: Joseph (or your choice)
   - Homepage URL: `http://localhost:8000`
   - Authorization callback URL: `http://localhost:8000/auth/callback`
4. Click "Register application"
5. Copy the Client ID
6. Generate a new Client Secret and copy it

### 2. Create Environment File

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your OAuth credentials:

```env
GITHUB_CLIENT_ID=your-actual-github-client-id
GITHUB_CLIENT_SECRET=your-actual-github-secret
JWT_SECRET_KEY=generate-a-random-secret-key-here
```

**Generate a secure JWT secret:**

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Start the Application

```bash
docker compose up --build
```

The application will:
- Start PostgreSQL database
- Create database tables automatically
- Start the FastAPI application on port 8000

### 4. Test Authentication

1. Navigate to: `http://localhost:8000/auth/login/github`
2. Authorize with your GitHub account
3. You'll receive JWT tokens in the response

### 5. Using the API

#### Get Access Token

After logging in via OAuth, you'll receive:

```json
{
  "access_token": "eyJhbGc...",
  "refresh_token": "eyJhbGc...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "name": "User Name"
  }
}
```

#### Make Authenticated Requests

Include the access token in the Authorization header:

```bash
curl -X POST http://localhost:8000/api/detect \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text to analyze"}'
```

#### Refresh Token

When access token expires (15 minutes by default):

```bash
curl -X POST http://localhost:8000/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "YOUR_REFRESH_TOKEN"}'
```

#### View Results History

```bash
curl http://localhost:8000/api/results \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Database Schema

### Users Table

- `id` - Primary key
- `email` - Unique email from OAuth
- `oauth_provider` - 'google' or 'github'
- `oauth_id` - Provider's user ID
- `name` - Display name
- `picture` - Profile picture URL
- `is_active` - Account status
- `created_at` - Account creation timestamp
- `updated_at` - Last update timestamp

### Results Table

- `id` - Primary key
- `user_id` - Foreign key to users table
- `text_analyzed` - Original text
- `human_probability` - Final human probability
- `ai_probability` - Final AI probability
- `prediction` - 'human' or 'ai'
- ML and entropy metrics (various columns)
- `created_at` - Detection timestamp

## Security Notes

- **HTTPS Required in Production**: Use HTTPS to protect tokens in transit
- **JWT Secret**: Keep `JWT_SECRET_KEY` secret and rotate periodically
- **OAuth Secrets**: Never commit `.env` file to version control
- **Token Expiration**: Access tokens expire in 15 minutes, refresh tokens in 30 days
- **Database**: Change default PostgreSQL password in production

## Troubleshooting

### OAuth Redirect Mismatch

Ensure the redirect URI in your OAuth app settings exactly matches:
`http://localhost:8000/auth/callback`

### Database Connection Error

Check that PostgreSQL is running:
```bash
docker compose ps
```

### Token Validation Failed

Ensure you're using the access token (not refresh token) in API requests.

## Production Deployment

For production deployment:

1. Update `OAUTH_REDIRECT_URI` to your domain
2. Add your domain to OAuth app authorized redirect URIs
3. Use strong `JWT_SECRET_KEY`
4. Change PostgreSQL password
5. Enable HTTPS
6. Consider adding rate limiting
7. Set up proper logging and monitoring
