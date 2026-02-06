# TruthLens: AI Content Risk Analyzer

🔍 **Multi-model probabilistic analysis of AI-generated content with explicit uncertainty handling**

---

## ⚠️ CRITICAL: What TruthLens Is (and Isn't)

### What TruthLens Does
- ✅ Provides **probabilistic risk analysis** using multiple detection signals
- ✅ Explicitly reports **UNCERTAIN** when evidence is ambiguous
- ✅ Shows transparent confidence scores and technical details
- ✅ Combines frequency analysis, semantic analysis, and metadata checking

### What TruthLens Does NOT Do
- ❌ Claim 100% accuracy (impossible with current technology)
- ❌ Provide legal proof of AI generation
- ❌ Detect all AI-generated content (photorealistic AI is often indistinguishable)
- ❌ Make definitive judgments where uncertainty exists

---

## 🎯 Why Uncertainty Is a Feature

Many "AI detectors" falsely claim high accuracy rates (90%+) that don't hold up in practice. TruthLens takes a different approach:

- **Honest about limitations**: Photorealistic AI (Midjourney V6, DALL-E 3, Stable Diffusion 3) can be indistinguishable from real photos
- **Conservative thresholds**: High bar for "AI_LIKELY" or "REAL_LIKELY" verdicts
- **Transparent confidence**: Shows when signals disagree
- **Educational**: Helps users understand the probabilistic nature of detection

---

## 🏗️ System Architecture

### Backend (Python FastAPI)
- **Image Analysis Pipeline**: FFT/DCT frequency analysis + semantic distribution + metadata
- **Text Analysis Pipeline**: Perplexity/burstiness detection + phishing analysis + misinformation risks
- **Fusion Algorithm**: Bayesian ensemble with uncertainty propagation

### Browser Extension (Chrome MV3)
- **4 Interaction Modes**: Full page scan, element inspector, snip mode, context menus
- **Floating Panel UI**: Modern glass morphism design
- **Restricted Page Handling**: Graceful degradation on chrome:// and other restricted pages

---

## 📋 Prerequisites

Before you start, make sure you have:

- **Python 3.10+** installed
- **Google Chrome** browser
- **Terminal/Command Prompt** access
- **Internet connection** for downloading dependencies

---

## 🚀 INSTALLATION GUIDE

### Step 1: Install Backend

#### 1.1 Navigate to Backend Directory
```bash
cd truthlens/backend
```

#### 1.2 Install Python Dependencies
```bash
pip install -r requirements.txt
```

Or if you get permission errors:
```bash
pip install --user -r requirements.txt
```

#### 1.3 Start the Backend Server
```bash
python main.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Keep this terminal window open!** The backend must be running for the extension to work.

---

### Step 2: Install Browser Extension

#### 2.1 Open Chrome Extensions Page
1. Open Google Chrome
2. Go to `chrome://extensions/`
3. Enable **Developer mode** (toggle in top right corner)

#### 2.2 Load Extension
1. Click **"Load unpacked"**
2. Navigate to the `truthlens/extension` folder
3. Click **"Select Folder"**

#### 2.3 Verify Installation
You should see:
- 🔍 TruthLens icon in your extensions toolbar
- No errors in the extension card

---

## 🎮 HOW TO USE

### Method 1: Full Page Scan
- **Click** the TruthLens icon → "Scan Full Page"
- **Keyboard**: `Ctrl+Shift+T` (Mac: `Cmd+Shift+T`)
- Analyzes all text and images on the page

### Method 2: Element Inspector
- **Click** the TruthLens icon → "Pick Element (Inspector)"
- **Keyboard**: `Ctrl+Shift+I` (Mac: `Cmd+Shift+I`)
- Hover over any element and click to analyze

### Method 3: Context Menu
- **Right-click** selected text → "TruthLens: Analyze selected text"
- **Right-click** an image → "TruthLens: Analyze this image"

### Method 4: Snip Mode (Coming in v1.1)
- Region selection for specific areas
- Currently shows placeholder

---

## 📊 Understanding Results

### Verdict Levels

| Verdict | Meaning | When Used |
|---------|---------|-----------|
| **AI_LIKELY** | High probability of AI generation | Score >75% AND confidence >60% |
| **UNCERTAIN** | Ambiguous signals or low confidence | Default for most edge cases |
| **REAL_LIKELY** | High probability of real content | Score <35% AND confidence >60% |

### Confidence Indicators

- **5 dots filled**: Very confident (signals agree)
- **3 dots filled**: Moderate confidence
- **1-2 dots filled**: Low confidence (signals conflict)

### Technical Signals Explained

#### For Images:
- **Frequency Artifacts (0-100%)**: FFT/DCT analysis detecting AI patterns in frequency domain
- **Semantic Drift (0-100%)**: Unusual feature distributions suggesting AI generation
- **Provenance Strength (0-100%)**: Metadata and EXIF analysis

#### For Text:
- **AI Generation (0-100%)**: Perplexity and burstiness analysis
- **Phishing Risk (0-100%)**: Urgency keywords, suspicious patterns
- **Misinformation Risk (0-100%)**: Emotional language, lack of sources

---

## 🔧 TROUBLESHOOTING

### Extension Not Working

**Problem**: "Analysis failed" error
- **Solution**: Make sure backend is running (`python main.py`)
- **Check**: Backend should be at `http://localhost:8000`
- **Test**: Open browser to `http://localhost:8000` - should see API info

**Problem**: Extension icon grayed out
- **Solution**: Reload the extension on `chrome://extensions/`
- **Try**: Click the puzzle icon, pin TruthLens

### Restricted Pages

**Problem**: "TruthLens cannot run on this page"
- **Explanation**: Browser security blocks extensions on:
  - `chrome://` pages
  - Chrome Web Store
  - Google login pages
- **Solution**: Use TruthLens on regular webpages

### Backend Errors

**Problem**: `ModuleNotFoundError: No module named 'fastapi'`
- **Solution**: Install dependencies again: `pip install -r requirements.txt`

**Problem**: `Address already in use`
- **Solution**: Port 8000 is occupied. Change port in `main.py`:
  ```python
  uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
  ```
  Then update extension `CONFIG.API_URL` in `scripts/content.js`

---

## 🧪 TESTING THE SYSTEM

### Test Image Analysis
1. Find an AI-generated image online (e.g., from Midjourney showcase)
2. Right-click → "TruthLens: Analyze this image"
3. Check if it detects AI characteristics

### Test Text Analysis
1. Generate text with ChatGPT
2. Paste into a webpage or document
3. Select the text → Right-click → "TruthLens: Analyze selected text"

### Expected Results
- **AI images with obvious artifacts**: AI_LIKELY (60-90%)
- **Photorealistic AI images**: Often UNCERTAIN (40-70%)
- **Real photos with camera metadata**: REAL_LIKELY (20-40%)
- **AI text (ChatGPT style)**: AI_LIKELY if >100 words

---

## 🎓 TECHNICAL DETAILS

### Image Analysis Stack
1. **FFT/DCT Frequency Analysis** (40% weight)
   - Detects periodic patterns from GANs
   - Identifies noise suppression from diffusion models
   - Finds unnatural smoothness in mid-frequencies

2. **Semantic Distribution Analysis** (35% weight)
   - Color variance and saturation analysis
   - Gradient smoothness detection
   - Feature distribution checks

3. **Provenance & Metadata** (25% weight)
   - EXIF camera data verification
   - AI software signature detection
   - File format analysis

### Text Analysis Stack
1. **AI Generation Detection**
   - Sentence length burstiness (AI = uniform, human = varied)
   - Vocabulary diversity patterns
   - AI-typical phrase detection

2. **Phishing Detection**
   - Urgency keyword analysis
   - Personal info request patterns
   - Suspicious URL checking

3. **Misinformation Risk**
   - Absolute language detection
   - Emotional manipulation patterns
   - Source citation analysis

### Fusion Algorithm
```python
ai_score = weighted_sum(signals)
variance = variance(signal_values)
confidence = 1.0 - min(variance * 2, 0.7)

if ai_score > 0.75 and confidence > 0.6:
    verdict = "AI_LIKELY"
elif ai_score < 0.35 and confidence > 0.6:
    verdict = "REAL_LIKELY"
else:
    verdict = "UNCERTAIN"
```

---

## 📈 KNOWN LIMITATIONS

### Image Detection
- **Photorealistic AI** (Midjourney V6, DALL-E 3): Often undetectable
- **Heavily edited photos**: May trigger false positives
- **Upscaled/compressed images**: Reduced accuracy
- **Screenshots**: Lose original metadata

### Text Detection
- **Short texts** (<50 words): Unreliable
- **Carefully edited AI text**: Can pass as human
- **Technical writing**: AI-like patterns even when human-written
- **Non-English**: Lower accuracy

### General
- **No training data**: Uses heuristics, not trained models
- **Evolving AI**: New generators may bypass detection
- **Adversarial examples**: Deliberate evasion possible

---

## 🔒 PRIVACY & SECURITY

- **No data storage**: Analysis happens in real-time, nothing saved
- **Local processing**: Frequency analysis runs client-side
- **API calls**: Only sends content for analysis, no tracking
- **No telemetry**: Extension doesn't phone home

---

## 🛠️ DEVELOPMENT

### Project Structure
```
truthlens/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── analyzers/
│   │   ├── image_analyzer.py   # Image analysis pipeline
│   │   ├── text_analyzer.py    # Text analysis pipeline
│   │   └── page_analyzer.py    # Page aggregation
│   └── requirements.txt
├── extension/
│   ├── manifest.json           # Extension configuration
│   ├── popup.html             # Extension popup
│   ├── scripts/
│   │   ├── content.js         # Page interaction
│   │   ├── background.js      # Service worker
│   │   └── popup.js           # Popup logic
│   ├── styles/
│   │   └── truthlens.css      # UI styles
│   └── icons/                 # Extension icons
└── docs/                      # Documentation
```

### Adding New Signals
1. Add detection method to respective analyzer
2. Update signal weights in fusion algorithm
3. Add UI display in `content.js` result panel
4. Document in README

### API Endpoints
- `POST /analyze/text` - Text analysis
- `POST /analyze/image` - Image upload analysis
- `POST /analyze/image/url` - Image URL analysis
- `POST /analyze/page` - Full page analysis

---

## 📝 LICENSE

This project is free and open source. Use at your own risk.

**Disclaimer**: TruthLens is a research tool and should not be used as definitive proof of AI generation in legal, academic, or professional contexts without additional verification.

---

## 🤝 CONTRIBUTING

Contributions welcome! Areas for improvement:
- Better ML models (when free APIs available)
- Additional detection signals
- UI/UX enhancements
- Documentation improvements

---

## 📞 SUPPORT

**Issues?** Check the Troubleshooting section above.

**Questions?** Remember: Uncertainty is a feature, not a bug!

---

## 🎯 Design Philosophy

> "It is better to be roughly right than precisely wrong."

TruthLens prioritizes **honest uncertainty** over **false precision**. In a world of AI content, knowing what we *don't* know is just as important as knowing what we do.

---

**Built with scientific integrity. Used with appropriate skepticism.**
