# Digit-classification-MNIST-FlaskUI-API-with-model-training

## Introduction
This project implements a simple AI model for image classification trained on the MNIST dataset. It includes a Flask web service that provides an API to interact with the AI model and logs results to a MySQL database. The application is containerized using Docker and deployable via Kubernetes.

## Requirements
- MySQL ( if you want to use Database Locally )
- Docker
- Kubernetes
- Python 3.10
- PyTorch, Flask, MySQL connector (installable via requirements.txt)

## Installation
1. Clone the repository:

git clone https://github.com/Deepak-7564/Digit-classification-MNIST-FlaskUI-API-with-model-training.git
cd Project/app


2. Install Python dependencies:

pip install -r requirements.txt

3. Change the credentials as per your requirements in the .env file

## Usage

Run the Flask application:
python app.py

###For API
API Endpoint:

POST /predict
Content-Type: application/json
username : username
password : password

{
"data": "base64 image data"
}

###For UI
a) /
b) /predictui

## Docker Deployment
Build the Docker image:

docker build -t my-python-app .

Run the Docker container:
docker-compose up



## Kubernetes Deployment
Apply the Kubernetes configurations:

kubectl apply -f deployment.yaml
kubectl apply -f service.yaml


## Troubleshooting
- If you encounter mysql related error, try to chnage credentials in env as per your mysql db.
- For any other issues, please open an issue on the GitHub repository.

## Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

