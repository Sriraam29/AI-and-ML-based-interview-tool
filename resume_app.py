import os
import fitz  
import numpy as np
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def compute_similarity(resume_text, job_desc):
    embeddings = model.encode([resume_text, job_desc])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return round(similarity[0][0] * 100, 2)  

# HTML template with inline CSS (for simplicity, or you can save as a separate file)
index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Job Match Analyzer</title>
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        /* Reset default styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* Body Styling */
        body {
            font-family: 'Poppins', sans-serif;
            background: black;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }

        /* Title */
        h2 {
            font-size: 26px;
            font-weight: 600;
            color: #fff;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 1px;
            animation: fadeIn 0.5s ease-in-out;
        }

        /* Form Container */
        form {
            background: white;
            padding: 25px;
            width: 350px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            animation: fadeIn 0.6s ease-in-out;
        }

        /* Input Fields */
        input, textarea {
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            border: 1px solid #ccc;
            border-radius: 6px;
            font-size: 16px;
            transition: all 0.3s ease;
        }

        input:focus, textarea:focus {
            border-color: #2980b9;
            outline: none;
            box-shadow: 0 0 8px rgba(41, 128, 185, 0.2);
        }

        /* File Input */
        input[type="file"] {
            background: #ecf0f1;
            cursor: pointer;
            padding: 10px;
        }

        /* Submit Button */
        button {
            background: #2980b9;
            color: white;
            padding: 12px;
            border: none;
            border-radius: 6px;
            font-size: 18px;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s ease;
        }

        button:hover {
            background: #2471a3;
            transform: scale(1.05);
        }

        /* Score Display */
        h3 {
            margin-top: 20px;
            font-size: 20px;
            color: #27ae60;
            font-weight: 600;
            animation: popUp 0.5s ease-in-out;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes popUp {
            from { transform: scale(0.8); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }

        /* Responsive Design */
        @media (max-width: 600px) {
            form {
                width: 90%;
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <h2>Resume Job Match Analyzer</h2>
    <form method="POST" enctype="multipart/form-data">
        <label for="resume">Upload Resume (PDF):</label>
        <input type="file" name="resume" id="resume" accept=".pdf" required><br><br>
        <label for="job_desc">Job Description:</label>
        <textarea name="job_desc" id="job_desc" rows="4" cols="50" required></textarea><br><br>
        <button type="submit">Analyze</button>
    </form>
    {% if score %}
        <h3>Match Score: {{ score }}%</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "resume" not in request.files:
            return "No file uploaded", 400
        file = request.files["resume"]
        job_desc = request.form["job_desc"]

        if file.filename == "" or job_desc.strip() == "":
            return "Invalid input", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        resume_text = extract_text_from_pdf(filepath)
        score = compute_similarity(resume_text, job_desc)

        return render_template_string(index_html, score=score)

    return render_template_string(index_html, score=None)

if __name__ == "__main__":
    app.run(port=5001,debug=True)