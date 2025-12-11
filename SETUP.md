# Databricks Configuration Setup

## Option 1: Using Streamlit Secrets (Recommended)

1. Open the file `.streamlit/secrets.toml`
2. Add your Databricks credentials:

```toml
[DATABRICKS]
host = "https://dbc-xxxxx-xxxx.cloud.databricks.com"
token = "your-personal-access-token-here"
model_uri = "models:/workspace.default.apexlogistics/1"
```

3. Save the file
4. Restart the Streamlit app

**Note:** The `secrets.toml` file is already in `.gitignore` so it won't be committed to git.

## Option 2: Using Environment Variables

Set environment variables before running Streamlit:

```bash
export DATABRICKS_HOST="https://dbc-xxxxx-xxxx.cloud.databricks.com"
export DATABRICKS_TOKEN="your-personal-access-token-here"
export MODEL_URI="models:/workspace.default.apexlogistics/1"

streamlit run app.py
```

## Getting Your Databricks Token

1. Go to your Databricks workspace
2. Click on your user icon (top right)
3. Select "User Settings"
4. Go to "Access Tokens" tab
5. Click "Generate New Token"
6. Copy the token (you'll only see it once!)

## Getting Your Databricks Host

Your Databricks host URL looks like:
- `https://dbc-xxxxx-xxxx.cloud.databricks.com`
- Or `https://your-workspace.cloud.databricks.com`

You can find it in your browser's address bar when you're logged into Databricks.

## Model URI

The default model URI is: `models:/workspace.default.apexlogistics/1`

If your model is in a different location, update the `model_uri` in the secrets file or environment variable.



