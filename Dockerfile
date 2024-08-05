# Use the official Python image from the Docker Hub
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code into the container
COPY . .

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]





#
# FROM --platform=linux/amd64 python:3.10.13-slim
#
# RUN pip install poetry \
#     && poetry config virtualenvs.create false
#
# WORKDIR /app
#
# COPY poetry.lock pyproject.toml /app/
#
# RUN poetry lock \
#     && poetry install --no-root --without dev
#
# COPY . .
#
# EXPOSE 8080
#
# ENTRYPOINT [ "python", "-m", "models.endpoint" ]