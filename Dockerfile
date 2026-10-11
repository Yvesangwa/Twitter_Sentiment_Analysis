FROM python:3.14-slim

WORKDIR /app

# Install required libraries
RUN pip install --no-cache-dir flask scikit-learn joblib numpy scipy gunicorn

# Copy Flask application
COPY app/ ./app/

# Copy trained model and vectorizer
COPY models/ ./models/

# Expose API port
EXPOSE 5000

# Start Flask API
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "app.app:app"]