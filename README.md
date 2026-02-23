# 🏥 MedGemma Conversational Health Assistant

> **AI-Powered Decision-Support Health Guidance for Non-Expert Patients**  
> Built for the Google MedGemma Impact Challenge — Kaggle Notebook

---

> ⚠️ **MEDICAL DISCLAIMER**: This tool is **NOT** a diagnostic system. It provides health guidance and decision-support only. Always consult a qualified healthcare professional for medical advice. In emergencies, call your local emergency number immediately (108 / 911 / 999 / 000).

---

## 📌 Overview

The MedGemma Conversational Health Assistant is a multi-turn, agentic AI system that helps non-expert patients understand their symptoms and medical documents through natural conversation. It combines Google's MedGemma-4B multimodal model with a LangGraph reasoning pipeline, RAG-enhanced medical knowledge retrieval, and an interactive Gradio interface — all running 100% offline on consumer hardware.

**The core problem it solves**: Over 4 billion people worldwide lack adequate healthcare access. Patients cannot understand complex medical reports, panic over symptoms they don't recognize, and delay care due to confusion and fear. Traditional AI assistants require expensive cloud connectivity and give generic, non-conversational answers.

---

## 🧱 Tech Stack

| Component | Technology |
|-----------|------------|
| **Base Model** | `google/medgemma-4b-it` (4-bit NF4 quantized) |
| **Agent Framework** | LangGraph (10-node reasoning graph) |
| **RAG / Retrieval** | LangChain + FAISS + sentence-transformers |
| **UI** | Gradio 4.x |
| **Quantization** | BitsAndBytes 4-bit (NF4 + double quantization) |
| **Hardware Target** | Kaggle T4 GPU (16 GB VRAM) |
| **Architecture** | Single-notebook, multi-node agentic pipeline |

---

## ✨ Key Features

- **Medical Image Understanding** — Accepts X-rays, lab reports, prescriptions, and discharge summaries; extracts and explains findings in plain language
- **Symptom Parsing & Categorization** — Detects symptoms across 8 clinical categories (cardiovascular, respiratory, neurological, gastrointestinal, musculoskeletal, dermatological, urological, general)
- **Emergency Detection** — Identifies red-flag symptoms (chest pain, stroke signs, severe bleeding, etc.) and immediately escalates with emergency guidance
- **Adaptive Follow-Up Questioning** — Dynamically generates targeted follow-up questions (duration, severity, character, triggers, medications, history, demographics) to iteratively build a complete clinical picture
- **RAG-Enhanced Reasoning** — Retrieves relevant medical knowledge from a curated FAISS index to ground responses in established clinical guidelines
- **Risk Stratification** — Classifies risk as Low / Medium / High with confidence scores
- **Patient-Friendly Reports** — Generates structured, jargon-free reports with actionable care recommendations
- **Safety Layer** — Filters diagnostic overreach language, appends mandatory disclaimers, and applies content safety checks
- **100% Offline** — No data ever leaves the device; full privacy preservation

---

## 🏗️ System Architecture

The pipeline is built as a **10-node LangGraph directed acyclic graph**:

```
Image Input
    ↓
[Node 1] Image Interpreter        — MedGemma multimodal analysis of uploaded medical documents
    ↓
[Node 2] Symptom Interpreter      — Parse & categorize raw symptom text, detect emergencies
    ↓
[Node 3] Context Builder          — Combine image findings + symptoms + conversation history
    ↓
[Node 4] RAG Retriever            — FAISS semantic search over medical knowledge base
    ↓
[Node 5] Clinical Reasoner        — MedGemma text inference for clinical assessment
    ↓
[Node 6] Follow-up Generator      — Produce targeted clarifying questions based on clarity score
    ↓
[Node 7] Response Integrator      — Merge all inputs into unified patient context
    ↓
[Node 8] Risk Classifier          — Assign Low / Medium / High risk with confidence
    ↓
[Node 9] Explanation Generator    — Generate plain-language patient explanation
    ↓
[Node 10] Care Suggestion Generator — Produce actionable recommended actions
    ↓
Final Report (JSON + Markdown)
```

---

## 🤖 Model Details

**Model**: `google/medgemma-4b-it`

MedGemma is Google's medical-domain multimodal language model. This project uses the instruction-tuned (`-it`) variant with **4-bit NF4 quantization** via BitsAndBytes, enabling it to run on a T4 GPU with approximately 2.1 GB VRAM usage (versus ~8 GB for full precision).

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

The model is loaded with `device_map='auto'` to support both GPU and CPU inference. A text-only fallback is included if the multimodal processor fails to load.

---

## 📚 Medical Knowledge Base (RAG)

The system uses a curated in-memory medical knowledge base covering:

- Cardiovascular: Chest pain assessment, hypertension stages, heart failure signs
- Respiratory: Fever management, pneumonia indicators, asthma vs. COPD
- Neurological: Headache red flags (SNOOP4 criteria), head injury assessment
- Gastrointestinal: Abdominal pain patterns
- Urological: UTI indicators
- Dermatological: Wound and infection signs
- Endocrine: Diabetes and blood glucose management
- Medications: Common drug interactions and safety

This knowledge is chunked using `RecursiveCharacterTextSplitter`, embedded with `sentence-transformers/all-MiniLM-L6-v2`, indexed in FAISS, and retrieved via semantic similarity search at inference time.

---

## 🔁 Conversation Flow

The `HealthSessionManager` orchestrates multi-turn sessions:

1. **Intake** — User uploads optional medical image and describes symptoms
2. **Pipeline Run** — Full 10-node LangGraph pipeline executes
3. **Clarity Scoring** — If clarity score < 65% and rounds < 4, enter follow-up mode
4. **Follow-Up** — System asks up to 2 targeted questions per round; user answers are appended to conversation history
5. **Final Report** — Once clarity threshold is met (or emergency detected), a structured JSON + Markdown report is generated

Maximum follow-up rounds: **4**  
Clarity threshold: **65%**

---

## 🛡️ Safety Architecture

The safety layer applies multiple protections:

- **Emergency Escalation**: Detects phrases like "chest pain", "difficulty breathing", "seizure", "stroke", etc. and immediately surfaces the emergency alert with local emergency numbers
- **Diagnostic Language Filter**: Replaces phrases like "you have" or "diagnosed with" with appropriately hedged language
- **Uncertainty Injection**: Adds epistemic hedges for high-risk assessments
- **Content Safety**: Regex-based filtering for harmful or hopeless language
- **Mandatory Disclaimer**: Every response is appended with the standard medical disclaimer

---

## 🖥️ UI — Gradio Interface

The Gradio interface is structured in two panels:

**Left Panel (Input)**
- Medical image upload (X-ray, lab report, prescription, discharge paper)
- Symptom description text box with example prompts
- Submit and Reset buttons
- Conversation chat history

**Right Panel (Output)**
- Live health guidance report with risk level indicator (🟢 Low / 🟡 Medium / 🔴 High)
- Possible health concerns list
- Recommended actions
- Patient-friendly explanation
- Image analysis findings
- Medical source citations
- Session summary and processing time

Example symptom prompts included in the UI:
- "I have a fever of 39°C, sore throat, and body aches for 2 days"
- "Chest pain that started this morning, feels like pressure, mild shortness of breath"
- "Severe headache for 3 days, dizziness when standing up, feeling very thirsty"
- "Stomach pain below belly button, painful urination, slight fever since yesterday"
- "Fell and hit my head 2 hours ago, now having headache and feeling confused"

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended: 16 GB VRAM) or CPU with 8+ GB RAM
- HuggingFace account with access to `google/medgemma-4b-it`

### Install Dependencies

```bash
pip install transformers>=4.40.0 accelerate>=0.27.0 bitsandbytes>=0.43.0 \
            langchain>=0.2.0 langchain-community>=0.2.0 langgraph>=0.1.0 \
            faiss-cpu sentence-transformers gradio>=4.31.0 \
            Pillow torch torchvision huggingface_hub peft einops timm
```

### Authenticate with HuggingFace

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")
```

On Kaggle, store your token as a secret named `HF_TOKEN` and retrieve it via:

```python
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')
```

### Run

Execute all notebook cells in order (Sections 1–18). The Gradio app launches in Section 18:

```python
app.launch(share=False, debug=True, show_error=True, max_threads=2, inline=True)
```

---

## 🧪 Evaluation Suite

Six built-in test cases cover the key clinical scenarios:

| ID | Scenario | Expected Risk | Expected Emergency |
|----|----------|--------------|-------------------|
| TC001 | Fever with body ache | Medium | No |
| TC002 | Chest pain + left arm radiation | High | Yes |
| TC003 | Dehydration signs | Medium | No |
| TC004 | Possible UTI | Medium | No |
| TC005 | Head injury with confusion | High | No |
| TC006 | Minor finger wound infection | Low | No |

Run with `run_evaluation()` to get pass/fail results, risk accuracy, emergency detection accuracy, follow-up question generation, and final report presence.

---

## 📱 Edge Deployment

The system is designed to be deployable outside of cloud infrastructure:

| Environment | Feasibility | Notes |
|-------------|-------------|-------|
| **Kaggle T4 GPU** | ✅ Primary target | ~2–5s response time |
| **16 GB RAM Laptop** | ✅ Recommended | CPU inference ~10–30s |
| **Apple M2/M3** | ✅ Metal acceleration | Good performance |
| **8 GB RAM Windows/Linux** | ⚠️ Slower | 4-bit enables this |
| **Android (Termux)** | ⚠️ With llama.cpp | 2–4 GB phones, GGUF conversion |
| **iOS** | ⚠️ CoreML conversion | Swift integration required |

**Memory Footprint:**

| Component | Memory |
|-----------|--------|
| MedGemma 4-bit model | ~2.1 GB VRAM |
| FAISS medical index | ~10 MB RAM |
| MiniLM embeddings | ~80 MB RAM |
| Application overhead | ~500 MB RAM |
| **Total minimum** | **~3 GB RAM** |

---

## 🌍 Impact & Roadmap

**Target Population**: 4+ billion underserved patients globally  
**Primary Use Cases**: Rural clinics, home health monitoring, elderly care, community health workers  
**Cost**: Zero API cost (fully local inference)  
**Privacy**: 100% — no data leaves the device  

**Future Roadmap:**
- Multilingual support (Hindi, Swahili, Spanish, Arabic)
- Voice input/output for low-literacy users
- Integration with wearable sensor data
- Community health worker dashboard
- Fine-tuning on local disease prevalence datasets
- WhatsApp/SMS bot interface for feature phones
- Progressive Web App for browser-based access

---

## 📁 Project Structure (Notebook Sections)

| Section | Description |
|---------|-------------|
| 1 | Setup & package installation |
| 2 | Library imports and GPU detection |
| 3 | HuggingFace authentication |
| 4 | MedGemma model loading (4-bit quantized) |
| 5 | Image understanding pipeline |
| 6 | Symptom intake module & emergency detection |
| 7 | RAG pipeline with medical knowledge base |
| 8 | Conversational follow-up engine |
| 9 | LangGraph state definition & MedGemma inference helper |
| 9b | LangGraph node definitions (Nodes 1–10) |
| 9c | LangGraph workflow compilation |
| 10 | Decision engine & session manager |
| 11 | Safety layer |
| 12 | Pipeline runner & report formatter |
| 13 | Gradio chat UI |
| 14 | Evaluation suite |
| 15 | Edge deployment notes |
| 16 | Competition writeup |
| 17 | Demo video script |
| 18 | App launch |

---

## 📜 License & Acknowledgements

- **MedGemma**: Google DeepMind — [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
- **LangChain / LangGraph**: LangChain, Inc.
- **FAISS**: Meta AI Research
- **Gradio**: Hugging Face
- **BitsAndBytes**: Tim Dettmers et al.

This project was built for the **Google MedGemma Impact Challenge** on Kaggle. It is intended for research and demonstration purposes only and is not a certified medical device.

---

*Built with ❤️ to make healthcare guidance accessible to everyone, everywhere.*
