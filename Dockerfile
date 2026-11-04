# ECHO demo image (used by the Hugging Face Space). One server on port 7860:
# FastAPI serves the API and the built React app. The Companion needs a chat provider
# (local Ollama by default), which is not available in the hosted demo.

FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/ backend/
COPY models/ models/
COPY experiments/results/ experiments/results/
COPY datasets/raw/dravidiancodemix/ datasets/raw/dravidiancodemix/
COPY --from=web /web/dist frontend/dist

# Writable location for the SQLite database inside the container.
ENV ECHO_DATABASE_PATH=/tmp/echo.sqlite3
WORKDIR /app/backend
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
