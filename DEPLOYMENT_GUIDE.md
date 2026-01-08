# SiteGround Deployment Guide - Pneumonia Detector Flask App

## ⚠️ Important Notice
**SiteGround shared hosting does NOT support Flask applications directly.** SiteGround primarily supports:
- PHP applications (WordPress, Joomla, etc.)
- Static HTML/CSS/JavaScript websites
- Node.js (on some plans)

## Alternative Deployment Options

### Option 1: Deploy to Heroku (RECOMMENDED - Free Tier Available)

1. **Install Heroku CLI**: Download from https://devcenter.heroku.com/articles/heroku-cli

2. **Create Required Files**:

**Create `requirements.txt`**:
```
Flask==2.3.2
tensorflow==2.13.0
keras==2.13.1
numpy==1.24.3
Pillow==10.0.0
gunicorn==21.2.0
```

**Create `Procfile`** (no extension):
```
web: gunicorn main:app
```

**Create `runtime.txt`**:
```
python-3.10.12
```

3. **Deploy Steps**:
```bash
# Login to Heroku
heroku login

# Create new app
heroku create pneumonia-detector-yourname

# Deploy
git init
git add .
git commit -m "Initial deployment"
git push heroku main

# Open your app
heroku open
```

**Note**: Your .h5 model files (PNmodel.h5, etc.) will be deployed with the app. If files exceed Heroku's 500MB limit, use Git LFS.

---

### Option 2: Deploy to PythonAnywhere (Flask-Friendly, Free Tier)

1. **Sign up**: https://www.pythonanywhere.com (Free tier available)

2. **Upload Files via Web Interface**:
   - Go to "Files" tab
   - Upload all your files including .h5 models
   - Keep the same folder structure

3. **Install Dependencies**:
   - Go to "Consoles" → Start a Bash console
   - Create virtual environment:
     ```bash
     mkvirtualenv --python=/usr/bin/python3.10 myenv
     pip install flask tensorflow keras numpy pillow
     ```

4. **Configure Web App**:
   - Go to "Web" tab → "Add a new web app"
   - Select Flask
   - Point to your `main.py` file
   - Set working directory to your app folder

5. **Reload and Test**

---

### Option 3: Deploy to Railway (Modern Alternative)

1. **Sign up**: https://railway.app (Free $5/month credit)

2. **Create `requirements.txt`** and **`Procfile`** (same as Heroku)

3. **Deploy via GitHub**:
   - Push code to GitHub
   - Connect Railway to your GitHub repo
   - Railway auto-detects Flask and deploys

---

## If You MUST Use SiteGround

### Convert to Static Site (Limited Functionality)

You can only host the **frontend** on SiteGround as a static demo, but it won't work without a backend. You'd need to:

1. **Use FileZilla to Upload Static Files**:
   - Host: `sftp://yoursite.com` or get from SiteGround cPanel
   - Username: Your cPanel username
   - Password: Your cPanel password
   - Port: 22 (SFTP) or 21 (FTP)

2. **Upload to `public_html` folder**:
   - Only upload `Home.html` and `styles.css`
   - This creates a non-functional demo

3. **For actual functionality**, deploy the Flask backend to one of the options above and update the form action in HTML:
   ```html
   <form action="https://your-heroku-app.herokuapp.com/submit" method="post">
   ```

---

## Recommended Workflow for FileZilla + Alternative Backend

### Step 1: Deploy Backend (Use Heroku/PythonAnywhere)
Follow Option 1 or 2 above to get your Flask app running.

### Step 2: Upload Static Assets to SiteGround via FileZilla

1. **Download FileZilla**: https://filezilla-project.org/

2. **Get SiteGround FTP Credentials**:
   - Login to SiteGround cPanel
   - Go to "Site Tools" → "Devs" → "FTP Accounts Manager"
   - Note your hostname, username, password

3. **Connect in FileZilla**:
   - File → Site Manager → New Site
   - Protocol: SFTP or FTP
   - Host: `yoursite.com` or `sftp.yoursite.com`
   - Username: From cPanel
   - Password: From cPanel
   - Port: 22 (SFTP) or 21 (FTP)
   - Click "Connect"

4. **Upload Files**:
   - Navigate to `public_html` folder on the right pane (remote)
   - Drag these files from left pane (local):
     - Any documentation/static pages you want to host
     - Sample X-ray images for demo
   - Do NOT upload Python files (.py, .h5) - they won't work on SiteGround

5. **Update HTML Form Action**:
   ```html
   <!-- In Home.html, change form action to your deployed backend -->
   <form action="https://your-app-name.herokuapp.com/submit" method="post">
   ```

---

## Best Solution Summary

**For a Masters Project Portfolio**:

1. **Backend**: Deploy Flask app to **Heroku** or **PythonAnywhere** (both free)
2. **Frontend**: Either:
   - Host everything on Heroku/PythonAnywhere (simpler)
   - Or create a landing page on SiteGround that links to your live app

3. **Portfolio Presentation**:
   ```
   SiteGround Site: Project description, screenshots, GitHub link
   Live App: Hosted on Heroku/PythonAnywhere
   GitHub Repo: Source code
   ```

---

## Quick Start - Full Deployment to Heroku

```bash
# 1. Navigate to your project
cd "C:\Users\Absol\OneDrive\Documents\GitHub\Pneumonia"

# 2. Create requirements.txt
echo Flask==2.3.2 > requirements.txt
echo tensorflow==2.13.0 >> requirements.txt
echo keras==2.13.1 >> requirements.txt
echo numpy==1.24.3 >> requirements.txt
echo Pillow==10.0.0 >> requirements.txt
echo gunicorn==21.2.0 >> requirements.txt

# 3. Create Procfile
echo web: gunicorn PneumoniaDetectorWebApp.main:app > Procfile

# 4. Create .gitignore
echo __pycache__/ > .gitignore
echo *.pyc >> .gitignore
echo .DS_Store >> .gitignore

# 5. Initialize git (if not already done)
git init
git add .
git commit -m "Prepare for Heroku deployment"

# 6. Create Heroku app
heroku create pneumonia-detector-yourname

# 7. Deploy
git push heroku main

# 8. Open app
heroku open
```

---

## Support

- **Heroku Docs**: https://devcenter.heroku.com/articles/getting-started-with-python
- **PythonAnywhere Help**: https://help.pythonanywhere.com/pages/Flask/
- **Flask Deployment**: https://flask.palletsprojects.com/en/2.3.x/deploying/

---

**Note**: Model files (.h5) are large. If deployment fails due to size:
- Use Git LFS: `git lfs track "*.h5"`
- Or upload models to cloud storage (AWS S3, Google Cloud Storage) and download them at runtime
