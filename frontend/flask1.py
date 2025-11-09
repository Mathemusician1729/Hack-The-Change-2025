from flask import Flask, jsonify, render_template
import random, datetime
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
    global current_time

    
    current_time = datetime.datetime.now()

    power_generated = 28

    data_values = 34

    live_int_value = 6

    moe = 52.4

    current_time += datetime.timedelta(hours=1)
    time = current_time.strftime('%H:%M')

    # simulate data
    return jsonify({
                    "powergen": random.randint(200,400),
                    'cost': round(random.uniform(0,1.5),2),
                    "moe": round(random.uniform(0,5),2),
                    "timestamp": time
                    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True)