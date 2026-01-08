# Titanic

Project for titanic passenger survival rate prediction.


This project runs a FastAPI backend and Streamlit frontend. Backend serves API on port 8000, frontend runs separately.

## Run the Application
**Install Dependencies**
```bash
# Requirements
pip install -r requirements.txt
```
**Terminal for backend**
```bash
cd src
uvicorn backend:app --reload --port 8000
```

**New terminal for frontend**
```bash
cd src
streamlit run frontend.py --server.port 8501
```

Or from project root:
```bash
streamlit run src/frontend.py --server.port 8501
```

## Backend Access
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

## Frontend Access
- UI: http://localhost:8501

## Stop Servers
Ctrl+C in each terminal.
