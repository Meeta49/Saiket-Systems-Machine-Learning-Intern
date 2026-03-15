# 🩺 Diabetes Predictor — Deployment Guide

## Files in this project
```
├── app.py              ← Streamlit web app
├── train_model.py      ← Script to train & save model
├── requirements.txt    ← Python dependencies
├── diabetes.csv        ← Your dataset (rename from "diabetes (1).csv")
├── model.pkl           ← Generated after running train_model.py
└── features.pkl        ← Generated after running train_model.py
```

---

## Step 1: Prepare your files locally

1. Rename your dataset file:
   ```
   diabetes (1).csv  →  diabetes.csv
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train and save the model:
   ```bash
   python train_model.py
   ```
   This creates `model.pkl` and `features.pkl`.

4. Test the app locally:
   ```bash
   streamlit run app.py
   ```
   Open http://localhost:8501 in your browser.

---

## Step 2: Push to GitHub

```bash
git add app.py train_model.py requirements.txt model.pkl features.pkl diabetes.csv
git commit -m "Add Streamlit deployment files"
git push origin main
```

---

## Step 3: Deploy on Streamlit Cloud (FREE)

1. Go to 👉 https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository**: `Meeta49/Saiket-Systems-Machine-Learning-Intern`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy!**

Your app will be live at a public URL like:
`https://meeta49-diabetes-predictor.streamlit.app`

---

## Notes
- Streamlit Cloud is completely **free** for public repos
- It auto-redeploys every time you push to GitHub
- No server setup needed
