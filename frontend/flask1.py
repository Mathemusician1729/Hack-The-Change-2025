from flask import Flask, jsonify, render_template
import random
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('page.html')


@app.route('/api/values')
def get_values():
    global live_int_value
    global data_values
    global power_generated
    global moe


    power_generated = 28

    data_values = 34

    live_int_value = 6

    moe = 52.4

    output = 4.2

    return jsonify({'value': live_int_value,
                    "powergen": power_generated,
                    "moe": moe})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)