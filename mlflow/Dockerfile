FROM python:3.11

# Install python package
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY basic_auth.ini /mlflow/auth/basic_auth.ini