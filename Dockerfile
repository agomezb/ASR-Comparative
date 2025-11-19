FROM python:3.11-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

# Expose the port JupyterLab runs on
EXPOSE 8888

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
