from flask import Flask, render_template_string, render_template, request
import plotly 
import io
import base64
import matplotlib.pyplot as plt

# plot
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

html = """

"""
app = Flask(__name__, template_folder="C:/Users/tranm/Hack-The-Change-2025")

@app.route("/")
def idx():
    return render_template("ui.html", name="Flask User")

# @app.route("/plot", methods=["POST"])
# def plot():
#     if 'csv' not in request.files:
#         return render_template_string()

if __name__ == '__main__':
    app.run(debug=True)

