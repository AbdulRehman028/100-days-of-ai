# Install Urdu Language Support for Tesseract OCR

To enable Urdu handwriting recognition, you need to install the Urdu language data files for Tesseract.

## Method 1: Download Urdu Language Data Files

1. **Download the Urdu language file:**
   - Go to: https://github.com/tesseract-ocr/tessdata/blob/main/urd.traineddata
   - Click "Download" button (or right-click "Raw" and save)

2. **Find your Tesseract installation:**
   - Default location: `C:\Program Files\Tesseract-OCR\tessdata\`
   - Or: `C:\Program Files (x86)\Tesseract-OCR\tessdata\`

3. **Copy the file:**
   - Place `urd.traineddata` into the `tessdata` folder
   - Full path should be: `C:\Program Files\Tesseract-OCR\tessdata\urd.traineddata`

4. **Restart the Flask app**

## Method 2: Download All Language Packs

1. Download from: https://github.com/tesseract-ocr/tessdata/archive/refs/heads/main.zip
2. Extract the zip file
3. Copy all `.traineddata` files to `C:\Program Files\Tesseract-OCR\tessdata\`

## Supported Languages

Once installed, the app will automatically try:
- **eng** - English
- **urd** - Urdu
- **ara** - Arabic
- **hin** - Hindi
- **eng+urd** - English + Urdu mixed

## Verify Installation

Run this command to check installed languages:
```powershell
tesseract --list-langs
```

You should see `urd` in the list.

## Alternative: Use Online OCR Services

If Tesseract doesn't work well for Urdu, consider using:
- Google Cloud Vision API (better for Urdu handwriting)
- Microsoft Azure Computer Vision
- AWS Textract

These services have better support for non-Latin scripts and handwriting recognition.
