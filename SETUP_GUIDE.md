# 🚀 DeepGuard - Complete Beginner's Setup Guide

This guide will help you run DeepGuard on a brand new computer, even if you've never coded before!

---

## 📋 What You Need First

Before we start, you need to install these programs (they're free!):

### Step 1: Install Python 🐍

1. Go to: https://www.python.org/downloads/
2. Click the big yellow **"Download Python 3.12"** button
3. Run the downloaded file
4. ⚠️ **IMPORTANT**: Check the box that says **"Add Python to PATH"**
5. Click "Install Now"
6. Wait for installation to complete

**To verify it worked:**
- Press `Windows + R`
- Type `cmd` and press Enter
- Type `python --version` and press Enter
- You should see something like `Python 3.12.x`

---

### Step 2: Install Node.js 📦

1. Go to: https://nodejs.org/
2. Click the **LTS** version (the one on the left)
3. Run the downloaded file
4. Click "Next" through all the screens
5. Click "Install"

**To verify it worked:**
- Open a new Command Prompt (Windows + R, type `cmd`)
- Type `node --version` and press Enter
- You should see something like `v20.x.x`

---

## 📂 Setting Up the Project

### Step 3: Open Command Prompt in Project Folder

1. Open File Explorer
2. Navigate to: `d:\projects\Deepfake detection`
3. Click on the address bar at the top
4. Type `cmd` and press Enter
5. A black Command Prompt window will open

---

### Step 4: Install Python Packages

In the Command Prompt, copy and paste this command, then press Enter:

```bash
pip install -r requirements.txt
```

⏳ This will take 5-10 minutes. You'll see lots of text scrolling. Wait until you see the cursor again.

---

### Step 5: Install Frontend Packages

In the same Command Prompt, type these commands one by one:

```bash
cd ui
npm install
```

⏳ This will take 2-5 minutes. Wait for it to finish.

---

## ▶️ Running the Application

You need to open **TWO** Command Prompt windows!

### Window 1: Start the Backend Server

1. Open Command Prompt in `d:\projects\Deepfake detection`
   - (File Explorer → go to folder → type `cmd` in address bar)
2. Run this command:

```bash
python scripts/server.py
```

✅ You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Leave this window open!**

---

### Window 2: Start the Frontend

1. Open a **NEW** Command Prompt in `d:\projects\Deepfake detection\ui`
   - (File Explorer → go to `ui` folder → type `cmd` in address bar)
2. Run this command:

```bash
npm run dev
```

✅ You should see:
```
  VITE v5.0.8  ready in XXX ms

  ➜  Local:   http://localhost:5173/
```

---

## 🎉 Open the App!

1. Open your web browser (Chrome, Edge, Firefox)
2. Type in the address bar: `http://localhost:5173`
3. Press Enter

**You should see the DeepGuard interface!**

---

## 🛑 How to Stop the Application

1. Go to each Command Prompt window
2. Press `Ctrl + C`
3. Type `Y` if it asks to terminate

---

## 🔧 Troubleshooting Common Issues

### "python is not recognized"
- You didn't check "Add Python to PATH" during installation
- Uninstall Python, reinstall, and make sure to check that box!

### "npm is not recognized"
- Close and reopen Command Prompt after installing Node.js
- If still not working, reinstall Node.js

### "No module named 'torch'"
- The pip install didn't complete properly
- Run `pip install -r requirements.txt` again

### Page won't load in browser
- Make sure BOTH Command Prompt windows are running
- Check that you typed the URL correctly: `http://localhost:5173`

### "Port already in use"
- Another program is using the port
- Close other applications or restart your computer

---

## 📱 Using the App

1. **Upload a Video**: Drag a video file (.mp4, .avi, .mov) into the upload area
2. **Use Webcam**: Click "Start Camera" for real-time detection
3. **Choose Mode**:
   - Fast = Quick scan, good for live video
   - Balanced = Best for most videos
   - Accuracy = Most thorough, takes longer
4. **View Results**: See the risk meter and explanation

---

## 🔄 Running Again Later

Every time you want to use the app:

1. Open Command Prompt in `d:\projects\Deepfake detection`
2. Run: `python scripts/server.py`
3. Open new Command Prompt in `d:\projects\Deepfake detection\ui`
4. Run: `npm run dev`
5. Open browser to: `http://localhost:5173`

---

**Need help?** Create an issue on the project repository! 🆘
