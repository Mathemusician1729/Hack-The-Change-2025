from flask import Flask, render_template_string
import matplotlib.pyplot as plt
import io
import base64

# make chart
fig, ax = plt.subplots()
x = [1, 2, 3, 4, 5]
y = [10, 20, 30, 40, 50]
ax.plot(x,y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Sample Plot')

# save chart
bf = io.BytesIO()
fig.savefig(bf, format='png')
plt.close(fig)
bf.seek(0)
img_base = base64.b64encode(bf.getvalue()).decode('utf-8')

html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Aura Prosumer Dashboard</title>
</head>
<body>
    <h1>Testy</h1>
<div>
    <h2>Energy Flow Visualization</h2>
    <img src="data:image/png;base64,{img_base}" alt="Energy Flow Graph">
</div>
</body>
</html>
"""

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template_string(html, name="Flask User")

if __name__ == '__main__':
    app.run(debug=True)

