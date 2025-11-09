from flask import Flask, render_template_string, render_template, request, redirect, url_for
import plotly 
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import torch

# load AI model
# model = torch.load()
# model.eval()

# mainpage html (the one that picks up csv)
html_title = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta chaset="UTF-8">
        <title>EnergyHub - Set Your Energy Goals!</title>
        <style>
            h1 {color: #0004ff; 
                font-size: 65px;
            }
            h2 {color: #0004ff; }
            h3 {color: #0004ff; }
            body {font-family: 'Inter', sans-serif; text-align: center; padding: 20px; background-color: #cad5e0; }
            p {color: #0004ff; }
        </style>
    </head>
    <body>
        <h1><b>Welcome To EnergyHub!</b></h1>
        <h2>Let's put your home energy to work.</h2>

        <!-- File Upload Form -->
        <form method="POST" action="/plot" enctype="multipart/form-data" class="space-y-4">
                <p>Please upload a .csv file for the data</p>
                <input 
                    type="file" 
                    name="csv_file" 
                    accept=".csv" 
                    required
                >
                <button type="submit">
                    Get Report
                </button>
            </form>
    </body>
</html>

"""

# dashboard html
html_dashboard = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta chaset="UTF-8">
        <title>EnergyHub - Set Your Energy Goals!</title>
        <style>
            h1 {color: #0004ff; 
                font-size: 65px;
            }
            h2 {color: #0004ff; }
            h3 {color: #0004ff; }
            body {font-family: 'Inter', sans-serif; text-align: center; padding: 20px; background-color: #cad5e0; }
            p {color: #0004ff; }
        </style>
    </head>

    <body>
        <h1><b>Get fuckin trolled lolololol</b></h1>
    </body>
</html>
"""

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template_string(html_title)

@app.route("/plot", methods=["POST"])
def plot():
    if 'csv_file' not in request.files:
        return render_template_string(html_title, error="No File Found")
    
    data = request.files['csv_file']
    if data.filename == "":
        return render_template_string(html_title, error="No File Selected")
    elif not data.filename.endswith(".csv"):
        return render_template_string(html_title, error="Invalid Filetype")
    try:
        df = pd.read_csv(data)
    except Exception as e:
        return render_template_string(html_title, error="Couldn't Read csv")
    
    # matplotlib code

    return redirect(url_for('dashboard'))

@app.route("/dashboard")
def dashboard():
        # display dashboard
        return render_template_string(html_dashboard)

if __name__ == '__main__':
    app.run(debug=True)

