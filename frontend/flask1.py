from flask import Flask, jsonify, render_template
import random
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('page.html')


@app.route('/api/values')
def get_values():
    global live_int_values
    global data_values
    global power_generated


    power_generated = 28

    data_values = 34

    live_int_value = 6

    output = 4.2

    return jsonify({'value': live_int_value,
                    "powergen": power_generated})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)