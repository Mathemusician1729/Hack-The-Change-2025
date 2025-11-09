from flask import Flask, jsonify, render_template
import os
import random
import json
import requests
from openai import OpenAI
app = Flask(__name__)
import datetime 

api_key = os.environ.get("OPENAI_API_KEY")

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

    
    current_time = datetime.datetime.now()

    power_generated = 28

    data_values = 34

    live_int_value = 6

    moe = 52.4

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
   
    return jsonify({'value': live_int_value,
                    "powergen": power_generated,
                    "moe": moe,
                    "AISummary": completion.choices[0].message.content})



if __name__ == '__main__':
    app.run(debug=True, threaded=True)