# CardiacZ - Heart Sound Analysis Application

A full-stack application for analyzing heart sounds using machine learning, built with FastAPI backend and React frontend.

## 🏗️ Project Structure

```
heart1/
├── backend/           # FastAPI Python backend
│   ├── main.py       # Main application entry point
│   ├── models/       # ML models and data
│   └── pyproject.toml # Python dependencies
├── frontend/         # React TypeScript frontend
│   ├── src/          # Source code
│   ├── package.json  # Node.js dependencies
│   └── vite.config.ts # Vite configuration
├── compose.yaml      # Docker Compose configuration
└── env.example       # Environment variables template
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Git

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd heart1
```

### 2. Environment Setup

Copy the environment template and configure your API keys:

```bash
cp env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY="your_openai_api_key_here"
```

### 3. Run with Docker

```bash
docker-compose up --build
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## 🛠️ Development Setup

### Backend Development

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install Python dependencies with Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

4. Run the development server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will be available at http://localhost:5173

## 📦 Dependencies

### Backend (Python)
- FastAPI - Web framework
- TensorFlow - Machine learning
- scikit-learn - ML utilities
- pandas - Data manipulation
- librosa - Audio processing
- OpenAI - AI integration

### Frontend (React/TypeScript)
- React 18
- TypeScript
- Vite - Build tool
- Tailwind CSS - Styling
- Recharts - Charts
- i18next - Internationalization
- Axios - HTTP client

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Docker Configuration

The application uses Docker Compose with two services:
- `backend`: FastAPI application on port 8000
- `frontend`: React application on port 3000

## 🧪 Testing

### Backend Tests
```bash
cd backend
poetry run pytest
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 📝 API Documentation

Once the backend is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🌐 Internationalization

The frontend supports multiple languages:
- English (en)
- Thai (th)

Language files are located in `frontend/src/locales/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions, please open an issue in the GitHub repository.
