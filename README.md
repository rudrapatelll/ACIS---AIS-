# Automated Component Inspection System (ACIS)

> An AI-powered component verification system for automotive assembly lines

**Author:** Rudra Patel  
**Student ID:** 20034606  
**Email:** patel.rudra@ufl.edu

---

##  Project Overview

ACIS is an AI-powered quality control system designed to automate component verification on automotive assembly lines. The system uses a hybrid approach combining Convolutional Neural Networks (CNNs) for classification and YOLO (You Only Look Once) for detection, built with TensorFlow, PyTorch, and OpenCV. This allows the system to both locate and verify component variants in real-time, instantly confirming whether the correct part is installed and eliminating manual inspection errors.

### Key Features
- **Dual AI approach:** CNN for classification + YOLO for detection (optional)
- **Real-time verification** of component variants (5-6 key components for proof-of-concept)
- **High accuracy** targeting ≥95% with controlled lighting conditions
- **Fast processing** with 200-300ms latency and 3-5 images/sec throughput
- **Custom dataset** collected from proprietary automotive plant data

### Model Architecture Options

**Option 1: CNN Classification (Primary)**
- Best for: Pre-cropped images with standardized ROI
- Architecture: Custom CNN or transfer learning (ResNet, EfficientNet)
- Use case: Component already centered in frame via fixed camera setup

**Option 2: YOLO Detection (Optional)**
- Best for: Locating components in full assembly line images
- Architecture: YOLOv8 for real-time object detection
- Use case: Variable component positions, multiple parts in single frame
- Can feed detected regions into CNN for fine-grained classification

### Project Scope
- Focus on **part-variant verification**, not cosmetic defect detection
- Proof-of-concept covers 5-6 critical dashboard components
- Designed for environments with consistent lighting and clear line of sight

---

##  Budget & Resources

### Training Platform
- **Primary:** Google Colab (free GPU access)
- **Local Storage:** Dataset stored on laptop (no cloud upload needed)
- **Cost:** $0 for training, testing, and running the model

### Optional Hardware
- **GPU Options:** NVIDIA RTX 4070 Ti or RTX 4050 for local development
- Budget-friendly alternatives available without sacrificing performance

---

##  Stakeholders

| Role | Responsibility |
|------|----------------|
| **Project Sponsors & Business Leaders** | Secure funding, measure business success |
| **Data Scientists & ML Engineers** | Build and train AI models |
| **Production Planning & Operations** | Balance efficiency and minimize downtime |
| **Data Engineers & Architects** | Set up data pipelines |
| **QA Manager** | Ensure final product quality |
| **Logistics & Inventory Teams** | Manage part availability |
| **End Users** | Automotive manufacturers and assembly plants |

---

##  Computing Infrastructure

### Performance Targets
- **Latency:** ≤ 200-300 ms
- **Throughput:** 3-5 images/sec (scalable)
- **Accuracy:** ≥ 95%
- **Uptime:** ≥ 99%

### Hardware Requirements

**Training & Inference:**
- **GPU:** NVIDIA RTX 4070 Ti (or equivalent)
- **Memory:** 16 GB RAM
- **Storage:** 300 GB SSD for datasets and model checkpoints

**Deployment:**
- **Current:** Cloud-based GPU servers (Google Cloud Platform)
- **Future:** Optional edge deployment for reduced latency

### Software Stack

**Operating Systems:**
- macOS (local development)
- Ubuntu 22.04 LTS (cloud training/inference)

**Frameworks & Libraries:**
- **AI Frameworks:** TensorFlow (CNN), YOLOv8 (optional detection)
- **Image Processing:** OpenCV
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Evaluation:** Scikit-learn
- **Containerization:** Docker

**Cloud Services:**
- **Provider:** Google Cloud Platform (GCP)
- **Training & Deployment:** Vertex AI
- **Database:** Cloud SQL
- **Storage:** Google Cloud Storage
- **Scaling:** Auto-scaling for inference endpoints

---

##  Security, Privacy & Ethics

### Problem Definition
- Structured stakeholder interviews (QA managers, assembly workers, supervisors)
- Manufacturing ethics framework prioritizing vehicle occupant safety
- Human oversight required for safety-critical components

### Data Collection
- **Privacy:** Differential privacy (Diffprivlib) and k-anonymity applied
- **Quality:** Standardized collection protocols (lighting, angles, resolution)
- **Bias Detection:** Fairlearn/AIF360 with ≤5% accuracy variance across groups

### Model Development
- **Robustness Testing:** Variable conveyor speeds, lighting schedules, component wear
- **Interpretability:** LIME/SHAP visualizations for automotive-specific decisions

### Deployment
- **Monitoring:** Real-time dashboards for confidence scores, processing time, error rates
- **Phased Rollout:** Shadow mode → Assisted mode → Autonomous mode (non-critical parts)

### Maintenance
- **Retraining:** Automatic triggers on 2% accuracy drop
- **Bias Reviews:** Monthly using Fairlearn/AIF360
- **Explainability:** Quarterly assessments with LIME/SHAP
- **Uncertainty Tracking:** Uncertainty Toolbox for flagging low-confidence predictions

---

##  Human-Computer Interaction (HCI)

### User Personas

**Persona 1: Jordan Alvarez (Primary)**
- Assembly operator, 32 years old
- Needs instant, unambiguous part-variant confirmation
- Works in fast-paced, variable lighting conditions
- Requires clear pass/fail feedback with minimal false alerts

**Persona 2: Priya Desai (Secondary)**
- QA manager, 41 years old
- Oversees multiple stations, validates exceptions
- Needs trend data and audit-ready logs
- Focuses on accuracy improvements without slowing production

### Usability Goals
- **Effectiveness:** Correct part selection with minimal errors
- **Efficiency:** Quick task completion with minimal steps
- **Satisfaction:** User trust and confidence in the system
- **Learnability:** Easy to learn and reliable
- **Accessibility:** WCAG 2.2 Level AA compliance

### Interface Design
- **Four station states:** Ready, Verifying, Pass, Mismatch
- **High-contrast feedback:** Color + text + audio cues
- **Touch-friendly:** Large targets for gloved use
- **Clear actions:** Re-seat & Recheck, Alert QA, Audited Override

### Development Process
1. **Wireframes:** Low-fidelity layouts in Figma
2. **Prototypes:** Interactive testing with real users
3. **Usability Testing:** System Usability Scale (SUS) questionnaires
4. **A/B Testing:** Google Optimize/Optimizely for variant comparison
5. **Accessibility:** Voice control, customizable display, keyboard navigation

### Post-Launch Monitoring
- Analytics tracking (Google Analytics, Heap)
- In-product micro-surveys and feedback widgets
- Continuous iteration based on user behavior and feedback

---

##  Risk Management

### Problem Definition Risks
- **Risk:** Misalignment with objectives, ethical concerns
- **Mitigation:** Stakeholder interviews, ethics framework, clear success metrics

### Data Collection Risks
- **Risk:** Data quality, bias, privacy breaches
- **Mitigation:** Privacy compliance (k-anonymity), automated quality checks, encrypted storage

### Model Development Risks
- **Risk:** Overfitting, bias amplification, poor generalization
- **Mitigation:** Robustness testing, 5-fold cross-validation, Fairlearn monitoring

### Deployment Risks
- **Risk:** Security breaches, integration issues
- **Mitigation:** Phased rollout, user controls, GCP auto-scaling, A/B testing

### Monitoring & Maintenance Risks
- **Risk:** Model drift, emerging security threats
- **Mitigation:** Monthly bias checks, quarterly LIME/SHAP analysis, GCP security

### Residual Risk Assessment

| Risk | Likelihood | Impact | Action |
|------|-----------|--------|--------|
| Algorithmic bias | Possible | Moderate | Monthly fairness testing & retraining |
| Model drift | Probable | High | Automatic retraining pipelines |
| Data privacy breach | Improbable | Low | Maintain encryption & audits |
| Hardware/network downtime | Possible | Moderate | Backup servers, cached inference |

---

##  Data Collection & Management

### Data Type
- **Format:** Unstructured (images and videos)

### Collection Method
- **Source:** Proprietary datasets from internal automotive plants
- **Method:** Manual collection, human-curated
- **Training Ingestion:** PyTorch DataLoader with batching and shuffling
- **Deployment Ingestion:** REST APIs for real-time, Kafka for batch processing

### Legal Compliance
- **Frameworks:** GDPR, ISO/IEC 27001, NIST cybersecurity guidelines
- **Anonymization:** Metadata (operator IDs, batch numbers) removed
- **Security:** Encrypted laptop (FileVault), password-protected Google Colab

### Data Ownership
- **Owner:** Rudra Patel (project author)
- **Plant Data:** Educational/research use only, not commercial
- **Access Control:** Password-protected laptop, 2FA on Google Colab

### Metadata Management
- **System:** CSV-based with Python Pandas
- **Content:** Data source, timestamp (generalized), format, component type, label
- **Naming:** `ComponentType_Timestamp_Station_Label.jpg`

### Data Versioning
- **Method:** Manual folder structure (raw, processed, augmented)
- **Version Control:** Git for code/metadata, local storage for large files
- **Documentation:** CHANGELOG for preprocessing steps and label updates

### Data Preprocessing

**Normalization:**
- Min-Max scaling to [0, 1] range
- Brightness filtering (30-200 average pixel value)

**Resizing:**
- All images standardized to 224x224 pixels
- Bicubic interpolation for quality preservation

**Feature Selection:**
- Region-of-interest (ROI) cropping
- Edge detection for component positioning

### Data Augmentation
- **Planned Techniques:** Horizontal flips, rotations (±15°), zoom (0.9-1.1x), brightness (±20%)
- **Expected Expansion:** Dataset from 500 to ~2,000 samples
- **Goal:** Improve model robustness and generalization to real-world variations

### Risk Mitigation

**Privacy Breaches:**
- FileVault encryption, 2FA, k-anonymity
- Goal: Zero privacy incidents

**Data Quality Degradation:**
- Automated quality checks (blur detection, brightness range)
- Goal: Reduce low-quality images to <2%

### Trustworthiness Strategies

**Data Anonymization:**
- K-anonymity for operator privacy
- Generalized timestamps
- Encrypted storage with 2FA
- Expected Outcome: Maintain worker privacy while preserving data utility

**Transparency & Traceability:**
- Detailed metadata in CSV
- Version control with Git
- CHANGELOG for all modifications
- Expected Outcome: Full reproducibility and accountability

---

##  Getting Started

### Prerequisites
```bash
- Python 3.8+
- TensorFlow 2.x / PyTorch 2.x
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/acis-project.git
cd acis-project

# Install dependencies
pip install -r requirements.txt

# For YOLO support
pip install ultralytics
```

### Usage

**CNN Classification Mode:**
```bash
# Train the CNN model
python train_cnn.py --dataset path/to/dataset --epochs 50

# Run CNN inference
python inference_cnn.py --model path/to/model --image path/to/image
```

**YOLO Detection Mode:**
```bash
# Train YOLO model
python train_yolo.py --dataset path/to/dataset --epochs 100

# Run YOLO inference
python inference_yolo.py --model path/to/model --image path/to/image
```

**Hybrid Mode (YOLO + CNN):**
```bash
# Run detection + classification pipeline
python inference_hybrid.py --yolo-model path/to/yolo --cnn-model path/to/cnn --image path/to/image

# Launch dashboard
python dashboard.py
```

---

## Performance Targets

- **Target Accuracy:** ≥ 95%
- **Expected Accuracy (with augmentation):** 96% based on similar projects
- **Target Latency:** 200-300 ms
- **Target Throughput:** 3-5 images/sec
- **Target Uptime:** ≥ 99%
- **Cross-Validation Goal:** Consistent performance across data splits

---

## Future Enhancements

- Expand to 35-40 dashboard components
- Implement synthetic data generation using GANs
- Deploy edge computing for reduced latency
- Add real-time monitoring dashboards (Prometheus, Grafana)
- Integrate advanced security audits and penetration testing
- Explore ensemble methods combining YOLO detection with multiple CNN classifiers
- Implement attention mechanisms for better interpretability

---

## License

This project is for educational and research purposes only. Proprietary plant data is not publicly shared.

---

##  Contact

**Rudra Patel**  
Email: patel.rudra@ufl.edu  
Student ID: 20034606

---

##  Acknowledgments

- University of Florida
- Automotive manufacturing partners for providing proprietary data
- Google Colab for free GPU access
- Open-source community for TensorFlow, OpenCV, and related libraries
