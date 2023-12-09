from flask import Flask, request, jsonify, render_template
from flask_httpauth import HTTPBasicAuth
import mysql.connector
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv
import socket

def get_ip_address():
    # hostname is used to get the IP address
    hostname = socket.gethostname()
    # IP address is found using the socket.gethostbyname() method
    ip_address = socket.gethostbyname(hostname)
    return ip_address

# Get the current IP address
current_ip_address = get_ip_address()
print("current_ip_address",current_ip_address)


load_dotenv()
app = Flask(__name__)
auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    if username in users and users[username] == password:
        return username
    

app_username = os.getenv("APP_USERNAME")
app_password = os.getenv("APP_PASSWORD")
cuda_device = os.getenv("CUDA_DEVICE")
model_path = os.getenv("MODEL_PATH")
mysql_host = os.getenv("MYSQL_HOST")
mysql_user = os.getenv("MYSQL_USER")
mysql_password = os.getenv("MYSQL_PASSWORD")
mysql_database = os.getenv("MYSQL_DATABASE")

users = {
    app_username: app_password
}


device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x



# Load your trained model
model = Net().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# Database setup
db = mysql.connector.connect(
    host=mysql_host,
    user=mysql_user,
    password=mysql_password
)

def create_database_and_table():
    cursor = db.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {mysql_database}")
    db.database = mysql_database
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        input_data TEXT NOT NULL,
        prediction VARCHAR(255) NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
    """)
    cursor.close()

create_database_and_table()





# Define the transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
@auth.login_required
def predict():
    # Extract and decode the image from the request
    image_data = request.json['data']
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # Apply the transformations
    image = transform(image)

    # Add an extra batch dimension since pytorch treats all inputs as batches
    image = image.unsqueeze(0)

    # Predict
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    # Log to MySQL database
    db.database = mysql_database
    cursor = db.cursor()
    sql = "INSERT INTO predictions (input_data, prediction) VALUES (%s, %s)"
    val = (image_data, str(predicted.cpu().numpy()))
    cursor.execute(sql, val)
    db.commit()

    return jsonify({'prediction': predicted.cpu().numpy().tolist()})



@app.route('/predictui', methods=['GET'])
@auth.login_required
def upload_file():
    return render_template('upload.html')

@app.route('/predictui', methods=['POST'])
@auth.login_required
def predict_ui():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'

    if file:
        # Load the image
        image = Image.open(file.stream)

        # Convert PNG to JPEG if necessary
        if image.format == 'PNG':
            image = image.convert('RGB')
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            image = Image.open(buffered)

        # Save the original image to a buffer for logging
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Apply transformations for prediction
        transformed_image = transform(image)
        transformed_image = transformed_image.unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            transformed_image = transformed_image.to(device)
            outputs = model(transformed_image)
            _, predicted = torch.max(outputs.data, 1)

        # Log to MySQL database
        db.database = mysql_database
        cursor = db.cursor()
        sql = "INSERT INTO predictions (input_data, prediction) VALUES (%s, %s)"
        val = (img_str, str(predicted.cpu().numpy()[0]))
        cursor.execute(sql, val)
        db.commit()

        return render_template('result.html', prediction=predicted.cpu().numpy()[0])




if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000',debug=True)