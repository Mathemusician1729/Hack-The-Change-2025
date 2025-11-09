from flask import Flask, jsonify, render_template
import os
import random
import json
import requests
from openai import OpenAI
app = Flask(__name__)
import datetime 
import pandas as pd
import torch
from model import eletric_neuralnet
#api_key = os.environ.get("OPENAI_API_KEY")
model = eletric_neuralnet()
model.to("cuda")

model.eval()
df = pd.read_csv("smart_grid_dataset.csv")
csv_last_d1 = df["Power Consumption (kW)"].iloc[-1]
csv_last_d2 = df["Power Factor"].iloc[-1]
csv_last_d3 = df["Humidity (%)"].iloc[-1]

model.state_dict(torch.load(r"model_checkpoint_epoch_30_0.003284123493358493.pt")["model_state_dict"])

inPT = torch.tensor(csv_last_d1, dtype=torch.float32)
inPT = inPT.unsqueeze(0).to("cuda")
live_int_value_out = model(inPT)
    
model.state_dict(torch.load(r"grid_to_humidity_10_0.0028516692109405994.pt")["model_state_dict"])

inPT = torch.tensor(csv_last_d3, dtype=torch.float32)
inPT = inPT.unsqueeze(0).to("cuda")
moe_out = model(inPT)


model.state_dict(torch.load(r"Grid_to_cost_10_0.0027483240701258183.pt")["model_state_dict"])

    
data_values_out =  model(moe_out)



@app.route('/')
def index():
    return render_template('page.html')


@app.route('/api/values')
def get_values():
    global live_int_value
    global data_values
    global power_generated
    global moe
    global current_time

    
    model = eletric_neuralnet()
    model.to("cuda")

    model.eval()
    df = pd.read_csv("smart_grid_dataset.csv")
    csv_last_d1 = df["Power Consumption (kW)"].iloc[-1]
    csv_last_d2 = df["Power Factor"].iloc[-1]
    csv_last_d3 = df["Humidity (%)"].iloc[-1]







    
    current_time = datetime.datetime.now()

    power_generated = 56


    
    data_values =  data_values_out

    if live_int_value_out != None:
        live_int_value = 5
    else:
        live_int_value = live_int_value_out
    
    moe = 5.36

    output = 4.2


    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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"
    )

    completion = client.chat.completions.create(
        model="google/gemini-2.5-flash-lite-preview-09-2025",
        messages=[{
            "role": "user",
            "content":f"Analyze these metrics and then give a simple summary based on this, one sentences maximum 12 words. Solar Power Generated for home {power_generated}, Power Consumption{live_int_value}, Cost Based Power Consumption and city Grid efficiency {data_values}, grid effeincy affected cost {output}, previous usage {0}. Then based on the current market value of the selling and buying give recomendation on wheather to sell or store power in a battery make sure it is in html format and avoid quotations and saying html. is format: Home is Generating More/Less Than Production <b>Sell or Over Consumption or store<b>. Cant be all only say one"
        }]
    )
   
    return jsonify({'cost': live_int_value,
                    "powergen": power_generated,
                    "moe": moe,
                    "AISummary": completion.choices[0].message.content})



if __name__ == '__main__':
    app.run(debug=True, threaded=True)