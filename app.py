import os
import cv2
import torch
import base64
import torch.nn as nn

from flask import Flask, render_template, request, jsonify
from torchvision import transforms, models
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

cnn_model = models.mobilenet_v2(weights=None)

cnn_model.classifier[1] = nn.Sequential(

    nn.Dropout(0.4),

    nn.Linear(
        cnn_model.last_channel,
        2
    )
)

cnn_model.load_state_dict(
    torch.load(
        "mobilenet_model.pth",
        map_location=device
    )
)

cnn_model = cnn_model.to(device)

cnn_model.eval()

print("MobileNetV2 Loaded")

xception_model = models.resnet18(weights=None)

xception_model.fc = nn.Sequential(

    nn.Dropout(0.4),

    nn.Linear(
        xception_model.fc.in_features,
        2
    )
)

xception_model.load_state_dict(
    torch.load(
        "xception_model.pth",
        map_location=device
    )
)

xception_model = xception_model.to(device)

xception_model.eval()

print("ResNet18 Loaded")


def get_model(model_name):

    if model_name == "cnn":
        return cnn_model

    return xception_model

@app.route('/')
def home():

    return render_template("index.html")

@app.route('/select')
def select():

    return render_template("select.html")

@app.route('/image')
def image_page():

    return render_template("upload_image.html")

@app.route('/video')
def video_page():

    return render_template("upload_video.html")

@app.route('/image_result')
def image_result():

    return render_template("image_result.html")

@app.route('/video_result')
def video_result():

    return render_template("video_result.html")

@app.route('/predict_image', methods=['POST'])
def predict_image():

    try:

        file = request.files['file']

        model_name = request.form.get(
            "model",
            "cnn"
        )

        if file.filename == "":

            return jsonify({
                "error": "No image selected"
            })

        image_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(image_path)

        model = get_model(model_name)

        image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = transform(
            image
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            outputs = model(image_tensor)

            probs = torch.softmax(
                outputs,
                dim=1
            )[0]

        fake = float(probs[0]) * 100

        real = float(probs[1]) * 100

        label = (
            "REAL"
            if real > fake
            else "FAKE"
        )

        if label == "REAL":

            reason = (
                "Natural facial patterns detected."
            )

        else:

            reason = (
                "Possible deepfake artifacts detected."
            )

        return jsonify({

            "label": label,

            "real": round(real, 2),

            "fake": round(fake, 2),

            "image": file.filename,

            "reason": reason,

            "model": model_name
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


@app.route('/predict_video', methods=['POST'])
def predict_video():

    try:

        file = request.files['file']

        model_name = request.form.get(
            "model",
            "cnn"
        )

        if file.filename == "":

            return jsonify({
                "error": "No video selected"
            })

        video_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(video_path)

        model = get_model(model_name)

        cap = cv2.VideoCapture(video_path)

        frames = []

        real_scores = []

        fake_scores = []

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if frame_count % 10 != 0:
                continue

            frame = cv2.resize(
                frame,
                (224, 224)
            )

            
            _, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            frame_base64 = base64.b64encode(
                buffer
            ).decode("utf-8")

            frames.append(frame_base64)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            pil_image = Image.fromarray(rgb)

            image_tensor = transform(
                pil_image
            ).unsqueeze(0).to(device)

            
            with torch.no_grad():

                outputs = model(image_tensor)

                probs = torch.softmax(
                    outputs,
                    dim=1
                )[0]

            fake_scores.append(
                float(probs[0])
            )

            real_scores.append(
                float(probs[1])
            )

            
            if len(frames) == 8:
                break

        cap.release()

       
        if len(real_scores) == 0:

            return jsonify({

                "label": "UNKNOWN",

                "real": 0,

                "fake": 0,

                "frames": [],

                "reason":
                "No valid frames found"
            })

      
        real = (
            sum(real_scores)
            /
            len(real_scores)
        ) * 100

        fake = (
            sum(fake_scores)
            /
            len(fake_scores)
        ) * 100

        label = (
            "REAL"
            if real > fake
            else "FAKE"
        )

        
        if label == "REAL":

            reason = (
                "Video appears natural."
            )

        else:

            reason = (
                "Frame inconsistencies detected."
            )

        return jsonify({

            "label": label,

            "real": round(real, 2),

            "fake": round(fake, 2),

            "frames": frames,

            "reason": reason,

            "model": model_name
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":

    app.run(debug=True)