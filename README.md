# Emergency Companion Chatbot

A lightweight conversational AI system fine-tuned on emergency medical scenarios to provide immediate first aid guidance and preparedness information during critical situations.

### Links:

- [Dataset](https://drive.google.com/drive/folders/1LlQiEaPZ6Z_Md_0-70s6I-5DuuvRELKZ?usp=sharing)
- [Model](https://drive.google.com/drive/folders/1BpC6xuwQ2oWaAU-bhy3uTPLv2qZ0yalT?usp=sharing)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/emergency-companion-chatbot/blob/main/Emergency_companion_Summative.ipynb)

## Dataset Description

**Source:** Custom emergency response dataset (`intents.json`) containing 56 medical emergency categories

**Access:** [Download from Google Drive](https://drive.google.com/drive/folders/1LlQiEaPZ6Z_Md_0-70s6I-5DuuvRELKZ?usp=sharing)

**Original Data:**

- Pattern-response pairs covering common emergencies (bleeding, choking, burns, fractures, poisoning, etc.)
- Manually curated responses based on standard first aid protocols

**Preprocessing Pipeline:**

1. **Smart Augmentation:** Generated grammatically diverse query variants (e.g., "How to treat burns?" → "What should I do for a burn?", "Emergency burn treatment")
2. **Text Cleaning:**
   - Fixed encoding artifacts and special characters
   - Normalized whitespace and punctuation
   - Filtered responses <30 characters for quality
3. **Dataset Expansion:**
   - Initial: ~500 samples
   - Final: 4,095 samples (2,250 unique patterns)
4. **Conversational Formatting:** Structured as `<|user|> query <|assistant|> response` for causal language modeling

**Final Statistics:**

- Training samples: 3,480
- Test samples: 615
- Train/test split: 85%/15% (stratified by emergency category)

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, CPU works but slower)
- 4GB+ RAM

### Option 1: Run in Google Colab (Recommended)

1. Access trained model: [Trained Model (Google Drive)](https://drive.google.com/drive/folders/1BpC6xuwQ2oWaAU-bhy3uTPLv2qZ0yalT?usp=sharing)

2. **Open the notebook in Colab**

   - Click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/emergency-companion-chatbot/blob/main/Emergency_companion_Summative.ipynb)

3. **Run all cells sequentially**

   - The notebook handles all dependencies automatically
   - GPU is enabled by default in Colab (Runtime → Change runtime type → T4 GPU)

4. **Access the trained model**

   - Model is automatically saved to: `/content/drive/MyDrive/Emergency-companion-chatbot/Model/emergency-gpt2-final`

### Use Gradio locally (Usin venv)

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/emergency-companion-chatbot.git
cd emergency-companion-chatbot
```

2. **Download the trained model**

   - Download from [Google Drive](https://drive.google.com/drive/folders/1BpC6xuwQ2oWaAU-bhy3uTPLv2qZ0yalT?usp=drive_link)
   - Extract it to `./Model/emergency-gpt2-final/`

3. **Create and activate virtual environment**

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Run the Gradio UI**

```bash
python app.py
```

6. **Access the interface**

   - Open browser at `http://localhost:7860`
   - Type emergency queries in the chat interface

7. **Deactivate virtual environment**

```bash
deactivate
```

---

## Example Queries & Responses

![Sample Chatbot Interactions](./assets/examples_screenshot.png)

### Quick Test Queries

Try these example inputs to test the chatbot:

- `"Someone is bleeding heavily from their arm"`
- `"What to do if someone is choking?"`
- `"severe burn help"`
- `"Someone fainted and won't wake up"`
- `"snake bite emergency"`
- `"How do I treat a sprain?"`
- `"Heart attack symptoms"`
- `"Treating a broken bone"`

## Dependencies

```txt
torch>=2.0.0
transformers>=4.35.0
gradio>=4.0.0
datasets>=2.14.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
evaluate>=0.4.0
accelerate>=0.24.0
```

**Install all at once:**

```bash
pip install -r requirements.txt
```

## Model Information

**Architecture:** DistilGPT-2 (distilled version of GPT-2)

**Parameters:** 81.9 million

**Training Configuration:**

- Learning rate: 1e-4
- Batch size: 16 (effective, with gradient accumulation)
- Epochs: 5
- Max sequence length: 256 tokens
- Optimizer: AdamW with weight decay 0.01
- Warmup steps: 200

**Performance Metrics:**

- Training Loss: 0.8730
- Validation Loss: 0.1136
- BLEU Score: 0.1827
- ROUGE-1: 0.3376
- ROUGE-2: 0.2518
- ROUGE-L: 0.2975

**Training Time:** ~45 minutes on NVIDIA T4 GPU

**Model Size:** ~330MB (saved checkpoint)

## Project Structure

```
emergency-companion-chatbot/
│
├── Dataset/
│   └── README.md                    # Raw emergency scenarios
│
├── Model/README.md                  # Model accessibility instructions
│
├── Notebook/
|    └── emergency_companion_model_Summative.ipynb       # Training notebook
|
├── Frontend/
|    └──app.py                           # Gradio UI application
|
├── requirements.txt                 # Python dependencies
|
└── README.md                        # This file
```

## Features

- **Real-time Response Generation:** Beam search with nucleus sampling for coherent outputs
- **Context-Aware:** Understands emergency query variations and reformulations
- **Medically Structured:** Responses follow standard first aid protocols
- **Lightweight:** Runs on CPU or GPU (330MB model size)
- **User-Friendly Interface:** Clean Gradio chat UI with example queries
- **Uses Colab:** No local setup required, run directly in browser and download the save model
- **Extensible:** Easy to fine-tune on additional medical scenarios

## **Technical Limitations:**

- Dataset size (3,480 samples) may not cover rare emergencies
- No real-time fact verification or medical guideline updates
- Generated responses may occasionally lack specificity
- Not validated by medical professionals for clinical accuracy

## 🔮 Future Improvements

- [ ] Integrate Retrieval-Augmented Generation (RAG) for updated medical guidelines
- [ ] Add multilingual support for non-English speakers
- [ ] Implement emergency service geolocation and auto-dialing
- [ ] Include image recognition for injury assessment
- [ ] Expand dataset to cover 200+ emergency scenarios
- [ ] Add voice interface for hands-free operation
- [ ] Medical expert validation and certification

**⭐ If this project helps you, please star the repository!**
