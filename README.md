# 🐾 Pet Breed Classifier - Complete Setup Guide

**For:** Running on ASUS TUF Laptop with RTX 4050 (6GB VRAM)
**Project:** Fine-grained Pet Breed Classification (37 classes)
**Model:** EfficientNetV2B0 with Transfer Learning
**Expected Total Time:** ~1.5 hours (mostly training)

---

## 📋 WHAT YOU'LL GET 🙌☢

By the end, you'll have:
- ✅ A trained model with ~90-95% accuracy
- ✅ Training curves (for thesis figures)
- ✅ Confusion matrix (for thesis figures)
- ✅ Per-class accuracy chart
- ✅ Sample predictions visualization
- ✅ Full classification report with precision/recall/F1

---

## ⚙️ PART 1: ONE-TIME SETUP (~30 minutes)

### Step 1: Install Python 3.10 or 3.11

**IMPORTANT:** Do NOT use Python 3.12+ — TensorFlow compatibility issues.

- Download: https://www.python.org/downloads/
- During installation, **CHECK "Add Python to PATH"**
- Verify:
  ```
  python --version
  ```
  Should show `Python 3.10.x` or `Python 3.11.x`

### Step 2: Install NVIDIA GPU Drivers + CUDA

Your RTX 4050 needs these to work with TensorFlow:

1. **Update NVIDIA drivers:**
   - Download GeForce Experience: https://www.nvidia.com/en-us/geforce/geforce-experience/
   - Update to latest driver

2. **Install CUDA Toolkit 11.8:**
   - Download: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Choose: Windows → x86_64 → 11 → exe (local)
   - Run installer, choose **Express Installation**

3. **Install cuDNN 8.6 for CUDA 11.8:**
   - Download: https://developer.nvidia.com/rdp/cudnn-archive
   - (Requires free NVIDIA developer account)
   - Extract the zip, copy contents to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`

### Step 3: Create Project Folder

Open **Command Prompt** (Windows key → type "cmd" → Enter):

```cmd
cd Desktop
mkdir pet_classifier
cd pet_classifier
```

### Step 4: Copy Project Files

Copy these files into the `pet_classifier` folder:
- `01_train_model.py`
- `02_evaluate_model.py`
- `requirements.txt`
- This `README.md`

### Step 5: Create Virtual Environment

```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your command prompt.

### Step 6: Install Dependencies

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

This takes ~5-10 minutes and downloads ~2GB.

### Step 7: Verify GPU Works

Still in the activated venv, run:

```cmd
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

**If you see** `GPUs: [PhysicalDevice(name='/physical_device:GPU:0'...)]` → ✅ GPU WORKS!
**If you see** `GPUs: []` → ❌ CUDA/cuDNN not installed correctly, redo Step 2.

---

## 🚀 PART 2: TRAINING THE MODEL (~45 minutes)

### Step 1: Make Sure Virtual Environment is Active

```cmd
cd Desktop\pet_classifier
venv\Scripts\activate
```

### Step 2: Run the Training Script

```cmd
python 01_train_model.py
```

### What to Expect

**First run:**
1. Downloads dataset (~800MB, takes 5-10 minutes depending on internet)
2. Builds model
3. **Phase 1 Training** (~15 minutes): Trains the classifier head with frozen base
   - You should see accuracy climb from ~3% to ~85-90% across 10 epochs
4. **Phase 2 Training** (~20-30 minutes): Fine-tunes the entire model
   - Accuracy should improve to 92-95%

### Laptop Usage Tips

- 🔌 **Keep laptop plugged in** (training uses a LOT of power)
- ❄️ **Use a cooling pad if possible** (GPU runs hot)
- 🚫 **Don't run other heavy apps** (games, video editing, etc.)
- 💡 **Can minimize window** — training runs fine in background
- ⏸️ **DO NOT close the terminal** until you see "🎉 TRAINING COMPLETE!"

### If Training Crashes

**Out of Memory (OOM) Error:**
- Open `01_train_model.py`
- Find the line: `BATCH_SIZE = 16`
- Change to: `BATCH_SIZE = 8`
- Re-run the script

**Other errors:** Screenshot the error and send it over for help.

---

## 📊 PART 3: GENERATE THESIS FIGURES (~5 minutes)

After training completes successfully:

```cmd
python 02_evaluate_model.py
```

This generates:
- `outputs/figures/confusion_matrix.png` → Use in Results section
- `outputs/figures/sample_predictions.png` → Use in Results section
- `outputs/figures/per_class_accuracy.png` → Use in Discussion section
- `outputs/figures/training_curves.png` → Use in Experiments section
- `outputs/classification_report.txt` → Use for Results tables
- `outputs/results_summary.txt` → Quick overview

---

## 📁 FINAL OUTPUT STRUCTURE

After everything runs, your folder will look like:

```
pet_classifier/
├── 01_train_model.py
├── 02_evaluate_model.py
├── requirements.txt
├── README.md
├── venv/
└── outputs/
    ├── models/
    │   ├── best_phase1.keras
    │   ├── best_final.keras
    │   └── final_model.keras      ← Main model file
    ├── figures/
    │   ├── training_curves.png
    │   ├── confusion_matrix.png
    │   ├── sample_predictions.png
    │   └── per_class_accuracy.png
    ├── class_names.txt
    ├── classification_report.txt
    └── results_summary.txt
```

---

## ❓ TROUBLESHOOTING

### "No module named 'tensorflow'"
- You forgot to activate the venv. Run: `venv\Scripts\activate`

### "Could not load dynamic library 'cudart64_110.dll'"
- CUDA not installed correctly. Redo Part 1 Step 2.

### "ResourceExhaustedError: OOM"
- Lower `BATCH_SIZE` in `01_train_model.py` from 16 → 8 → 4

### Training stuck at low accuracy (< 50%)
- Check GPU is actually being used (Task Manager → Performance → GPU)
- If GPU usage is 0%, CUDA setup is broken

### "Kernel died" or laptop freezes
- Too much VRAM usage. Lower BATCH_SIZE.
- Close Chrome/browser tabs before training.

### Dataset download fails
- Check internet connection
- May need to delete `C:\Users\YourName\tensorflow_datasets` and retry

---

## 📬 WHAT TO SEND BACK AFTER TRAINING

Send these to the team:
1. Screenshot of the final output showing accuracy
2. All 4 PNG files from `outputs/figures/`
3. `outputs/classification_report.txt`
4. `outputs/results_summary.txt`

The actual model file (`final_model.keras`) is large (~30MB) — only send if needed.

---

## 📝 QUICK RUN CHEAT SHEET

Once setup is done, running again is simple:

```cmd
cd Desktop\pet_classifier
venv\Scripts\activate
python 01_train_model.py
python 02_evaluate_model.py
```

That's it. Good luck! 🚀
