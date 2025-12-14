# 📥 Tesseract OCR Installation Guide

## Quick Installation (Recommended)

### Option 1: Direct Download
1. Download Tesseract installer: https://github.com/UB-Mannheim/tesseract/wiki
2. Click on: **tesseract-ocr-w64-setup-5.3.3.20231005.exe** (or latest version)
3. Run the installer
4. **IMPORTANT**: During installation, note the installation path (default: `C:\Program Files\Tesseract-OCR`)
5. Complete the installation
6. Restart your Flask app

### Option 2: Using winget (Windows Package Manager)
```powershell
winget install --id UB-Mannheim.TesseractOCR
```

### Option 3: Using Scoop
```powershell
scoop install tesseract
```

## After Installation

1. **Verify Installation:**
```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

2. **Restart Flask App:**
```powershell
cd "Day-38-Handwriting-to-Text"
python app.py
```

3. **You should see:**
```
✅ Tesseract OCR available!
🤖 Using Real OCR Recognition
```

## Manual Download Links

- **Official GitHub**: https://github.com/UB-Mannheim/tesseract/wiki
- **Direct Download (64-bit)**: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe

## Troubleshooting

**If Tesseract is not detected after installation:**

1. Check installation path exists:
```powershell
Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

2. If installed in different location, update `app.py`:
```python
tesseract_path = r'YOUR_ACTUAL_PATH\tesseract.exe'
```

3. Common alternative paths:
- `C:\Users\<YourName>\AppData\Local\Programs\Tesseract-OCR\tesseract.exe`
- `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`

## Without Tesseract

The app will work in **simulation mode** using the LLM, but won't read actual image content.
To use real OCR, Tesseract installation is required.

---

**Current Status**: App will work in simulation mode until Tesseract is installed.
**After Tesseract**: App will use real OCR to read your handwriting images! ✍️
