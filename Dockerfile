FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Create user with specific UID (e.g., 10001) and no login access
RUN adduser --uid 10001 --disabled-password --gecos "" appuser

# Switch to that user
USER 10001

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
