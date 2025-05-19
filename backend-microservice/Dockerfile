# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app/src/backend_microservice

# Install Poetry
RUN pip install poetry==2.1.3

# Copy source code
COPY . /app

# Install dependencies
RUN poetry config virtualenvs.create false && poetry install --only main

# Expose port
EXPOSE 8000

# Run the application
CMD ["poetry", "run", "uvicorn", "backend_microservice.main:app", "--host", "0.0.0.0", "--port", "8000"]
# ENTRYPOINT [ "/bin/bash" ]
