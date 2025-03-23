import os
from emopred.prediction import predict_emotion

from flask import Flask, render_template, request, url_for

app = Flask(__name__)


root_path = os.path.dirname(os.path.abspath(__file__))
upload_folder = os.path.join(root_path, 'static', 'videos')
output_folder = os.path.join(root_path, 'static', 'outputs')

app.config['upload_folder'] = upload_folder
app.config['output_folder'] = output_folder


# routes
@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        vid = request.files['video']
        vid.save(os.path.join(app.config['upload_folder'], 'video_test.mp4'))
        preds = predict_emotion(os.path.join(app.config['upload_folder'], 'video_test.mp4'),
                                os.path.join(app.config['output_folder']))
        return render_template('display.html', preds=preds)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
