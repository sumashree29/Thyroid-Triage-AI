# 🏥 Thyroid Triage AI

**Intelligent Clinical Decision Support System for Automated Thyroid Disease Screening**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Overview

Thyroid Triage AI is a multi-agent AI system that automates thyroid disease patient triaging using machine learning and clinical pattern recognition. It analyzes multiple thyroid hormones (TSH, T3, T4, FTI) to provide instant risk assessment, triage prioritization, and evidence-based recommendations.

### ✨ Key Features

- 🤖 **Multi-Agent Architecture**: 4 specialized AI agents (Risk Scorer, Retriever, Reasoner, Summarizer)
- 🧠 **Hybrid AI**: Combines Random Forest ML (87% accuracy) with rule-based clinical pattern recognition
- 📊 **Multi-Hormone Analysis**: Analyzes TSH, T3, T4, and FTI together (not just TSH)
- 👥 **Dual Perspectives**: Generates tailored reports for both patients and doctors
- ⚡ **Real-Time**: < 2 second analysis per patient
- 📚 **Evidence-Based**: Backed by ATA 2016 and Endocrine Society guidelines

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/thyroid-agentic-ai.git
cd thyroid-agentic-ai

# Install dependencies
pip install -r requirements.txt

# Run the application
python api.py

# Open browser
# Navigate to http://localhost:8000
```

## 🎯 How It Works

```
Patient Input → Multi-Agent System → Risk Assessment → Triage Priority → Personalized Report
                ├─ Risk Scorer   (ML + Clinical Rules)
                ├─ Retriever     (Clinical Guidelines)
                ├─ Reasoner      (Evidence Linking)
                └─ Summarizer    (Report Generation)
```

## 📊 Example

**Input:**
```json
{
  "age": 52,
  "sex": "F",
  "tsh": 8.5,
  "t4": 65,
  "fti": 55
}
```

**Output:**
- Risk Score: **88%**
- Priority: **URGENT**
- Pattern: Primary Hypothyroidism
- Recommendation: Urgent endocrine referral

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI, Pydantic
- **ML**: scikit-learn (Random Forest), pandas, numpy
- **RAG**: ChromaDB, LangChain
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Data**: 3,774 patient thyroid records

## 📁 Project Structure

```
thyroid-agentic-ai/
├── api.py                    # FastAPI application
├── src/
│   ├── agents/              # 4 AI agents
│   │   ├── risk_scoring.py
│   │   ├── retriever.py
│   │   ├── reasoner.py
│   │   └── summarizer.py
│   └── core/
│       └── workflow.py      # Multi-agent orchestrator
├── models/
│   ├── risk_classifier.pkl  # Trained Random Forest
│   └── encoder.pkl          # Feature preprocessor
├── static/                  # Web UI
└── data/                    # Training data
```

## 🎨 Screenshots

### Main Interface
![Thyroid Triage AI Interface](https://github.com/user-attachments/assets/96aa5491-d328-4951-aac0-e5c0c46990cb)


### Risk Assessment
![Risk Assessment](https://github.com/user-attachments/assets/3b80b1c1-37ea-4401-870c-289fc4fd93de)


## 📈 Performance

- **ML Accuracy**: 87% (validation)
- **Processing Time**: < 2 seconds
- **Training Data**: 3,774 patients
- **Confidence Scoring**: ✅
- **Triage Categories**: URGENT, HIGH_PRIORITY, ROUTINE

## 🔬 Clinical Accuracy

The system uses clinical thresholds from:
- American Thyroid Association (ATA) 2016 Guidelines
- Endocrine Society Clinical Practice Guidelines
- Evidence-based medicine protocols

**Normal Ranges:**
- TSH: 0.45-4.5 mIU/L
- T3: 0.8-2.0 ng/dL
- T4: 70-150 μg/dL
- FTI: 70-150

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Devasish**
- GitHub: [@devasish1403](https://github.com/devasish1403)

- LinkedIn: [Adigoppula Devasish](https://linkedin.com/in/adigoppula-devasish)

**Sumashree Dornala**
- GitHub: [@sumashree29](https://github.com/sumashree29)
  
- LinkedIn: [Sumashree Dornala](www.linkedin.com/in/sumashree-dornala)

## 🙏 Acknowledgments

- Dataset: UCI Machine Learning Repository
- Clinical Guidelines: American Thyroid Association
- Inspiration: Improving healthcare accessibility through AI

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact [devasish1403@gmail.com](devasish1403@gmail.com)

---

⭐ **Star this repository if you find it helpful!**
