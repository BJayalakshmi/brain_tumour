# 🧠 BrainTumorAI — MRI Brain Tumor Detection with Grad-CAM

A full-stack web application that detects brain tumors from MRI scans using deep learning, generates Grad-CAM heatmaps to explain AI decisions, and provides personalized dietary recommendations — all wrapped in a modern, responsive UI.

---

## 📋 Table of Contents

- [Demo](#-demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [API Reference](#-api-reference)
- [Model Architecture](#-model-architecture)
- [Database Schema](#-database-schema)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Demo

| Feature | Description |
|---|---|
| Upload MRI | Drag & drop or click to upload a brain MRI image |
| AI Detection | Get TUMOR / NO TUMOR prediction in seconds |
| Grad-CAM | Visual heatmap showing which region influenced the decision |
| Risk Level | Plain-language result with actionable next steps |
| Scan History | All previous scans stored per user |

---

## ✨ Features

- **AI-Powered Detection** — YOLO11n-based classifier trained on brain MRI images
- **Grad-CAM++ Heatmaps** — Visual explanation of model decisions (red = tumor region, blue = normal)
- **Risk Level System** — 6 levels from 🔴 High Concern to 🟢 Clear Scan
- **Plain-Language Results** — Descriptions and next steps written for non-medical users
- **User Authentication** — Secure signup/login with bcrypt password hashing
- **Scan History** — Per-user history of all past scans with overlay thumbnails
- **Dietary Recommendations** — Brain-health nutrition guidance shown on dashboard
- **Responsive Design** — Dark medical-tech UI built with React

---

## 🛠 Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| React 18 | UI framework |
| React Router v6 | Client-side routing |
| CSS Variables | Theming and design system |
| Fetch API | Backend communication |
| Syne + DM Sans | Typography (Google Fonts) |

### Backend
| Technology | Purpose |
|---|---|
| Flask | Web framework |
| Flask-CORS | Cross-origin requests |
| Flask-SQLAlchemy | ORM for database |
| SQLite | Lightweight database (`brain.db`) |
| bcrypt | Password hashing |
| PyTorch | Deep learning inference |
| OpenCV | Image processing |
| Matplotlib | Grad-CAM visualization |
| Torchvision | Image transforms |
| Pillow | Image loading |

---

## 📁 Project Structure

```
brain_tumor/
│
├── backend/
│   ├── app.py                  # Flask app entry point
│   ├── database.py             # SQLAlchemy db instance
│   ├── models.py               # User & Prediction models
│   ├── gradcam.py              # Grad-CAM++ heatmap generation
│   ├── model_new.py            # YOLO11n model architecture
│   ├── evaluate.py             # Model evaluation script
│   ├── best_classifier.pt      # Trained model weights
│   ├── brain.db                # SQLite database (auto-created)
│   ├── brain_tumour/
│   │   └── uploads/            # Uploaded MRI + overlay images
│   └── routes/
│       ├── __init__.py
│       ├── auth.py             # /signup, /login
│       ├── predict.py          # /predict
│       └── history.py          # /history/:user_id
│
└── frontend/
    ├── public/
    └── src/
        ├── api.js              # Central API config & helpers
        ├── App.js              # Router setup
        ├── App.css             # Global styles & CSS variables
        ├── components/
        │   ├── Navbar.js
        │   ├── Navbar.css
        │   └── ProtectedRoute.js
        └── pages/
            ├── Home.js
            ├── Home.css
            ├── Login.js
            ├── Signup.js
            ├── AuthPages.css
            ├── Dashboard.js
            └── Dashboard.css
```

---

## 🔬 How It Works

### 1. Image Upload & Preprocessing
The uploaded MRI is resized to 224×224 and normalized using ImageNet statistics before being fed to the model.

### 2. Model Inference
YOLO11n (width_mult=0.5) processes the image through convolutional layers and outputs a single logit value, which is converted to a probability via sigmoid.

### 3. Prediction
```
prob > 0.5  →  TUMOR
prob ≤ 0.5  →  NO TUMOR
```

### 4. Grad-CAM++ Heatmap
Backpropagation through the final convolutional layer generates a spatial attention map, upsampled and overlaid on the original MRI. Red/orange = high activation (tumor), Blue = low activation (normal).

### 5. Risk Classification
| Prediction | Confidence | Risk Level |
|---|---|---|
| TUMOR | ≥ 90% | 🔴 High Concern |
| TUMOR | 70–89% | 🟠 Moderate Concern |
| TUMOR | 50–69% | 🟡 Low Concern |
| NO TUMOR | ≤ 15% | 🟢 Clear Scan |
| NO TUMOR | 16–30% | 🟢 Likely Normal |
| NO TUMOR | 31–49% | 🟡 Inconclusive |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+
- pip
- npm or yarn
- Git

---

### Backend Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-ai.git
cd brain-tumor-ai/backend
```

**2. Create and activate a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install flask flask-cors flask-sqlalchemy bcrypt torch torchvision pillow opencv-python matplotlib werkzeug
```

**4. Place your trained model weights**

Copy `best_classifier.pt` into the `backend/` folder:
```
backend/
└── best_classifier.pt   ← place here
```

**5. Run the backend**
```bash
python app.py
```

The backend will start at `http://127.0.0.1:5000`.  
The database (`brain.db`) and upload folder are created automatically on first run.

---

### Frontend Setup

**1. Navigate to the frontend folder**
```bash
cd ../frontend
```

**2. Install dependencies**
```bash
npm install
```

**3. Configure the API base URL** (if needed)

Open `src/api.js` and confirm:
```javascript
export const BASE_URL = "http://127.0.0.1:5000";
```

**4. Start the development server**
```bash
npm start
```

The app will open at `http://localhost:3000`.

---

## 📡 API Reference

### Auth

| Method | Endpoint | Body | Response |
|---|---|---|---|
| POST | `/signup` | `{ username, email, password }` | `{ msg: "User created" }` |
| POST | `/login` | `{ email, password }` | `{ msg, user_id, username }` |

### Predict

| Method | Endpoint | Body | Response |
|---|---|---|---|
| POST | `/predict` | FormData: `file`, `user_id` | `{ prediction, confidence, overlay, overlay_image }` |

### History

| Method | Endpoint | Response |
|---|---|---|
| GET | `/history/:user_id` | `[{ file, result, overlay, date }]` |

### Serve Images

| Method | Endpoint | Response |
|---|---|---|
| GET | `/brain_tumour/uploads/<filename>` | Image file |

---

## 🧠 Model Architecture

The model is a **YOLO11n** backbone repurposed as a binary image classifier:

```
Input: 224×224×3 MRI image
  ↓
Conv layers (feature extraction)
  ↓
Grad-CAM hooks registered on final conv layer
  ↓
Global Average Pooling
  ↓
Fully connected → single logit
  ↓
Sigmoid → probability (0–1)
  ↓
Threshold @ 0.5 → TUMOR / NO TUMOR
```

- `num_classes=1` — binary classification
- `width_mult=0.5` — half-width for faster inference
- Pretrained on ImageNet, fine-tuned on brain MRI dataset

---

## 🗄 Database Schema

### `users` table
| Column | Type | Notes |
|---|---|---|
| id | Integer | Primary key, auto-increment |
| username | String | Derived from email prefix |
| email | String | Unique |
| password | String | bcrypt hashed |

### `predictions` table
| Column | Type | Notes |
|---|---|---|
| id | Integer | Primary key, auto-increment |
| user_id | Integer | Foreign key → users.id |
| filename | String | Original MRI filename |
| result | String | `"TUMOR"` or `"NO TUMOR"` |
| overlay | String | Grad-CAM output filename |
| date | DateTime | Timestamp of scan |

---

## ⚠️ Medical Disclaimer

> This application is an **AI-assisted screening tool only** and does **not** constitute a medical diagnosis. The predictions made by this model should never replace professional medical advice, diagnosis, or treatment. Always consult a qualified medical professional for any health concerns.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Built as a final year project for AI-powered medical imaging.  
Feel free to reach out for questions or collaborations.
