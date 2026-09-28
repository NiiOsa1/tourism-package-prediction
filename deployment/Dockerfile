# Base image: minimal Linux with Python 3.11 pre-installed
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the deployment folder into the container
COPY . .

# Install the Python packages listed in requirements.txt
RUN pip3 install -r requirements.txt

# Create a non-root user for security (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the app directory for the non-root user
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Launch the Streamlit app on port 7860 (HF Spaces default for Docker)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
