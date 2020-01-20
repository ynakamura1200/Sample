
import os
from io import BytesIO
from flask import Flask, request, redirect, url_for, render_template, send_file, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np

from app import app
from PlotTrainningResults import plot

import TrainningExecution as train

@app.route("/plot/")
def MyDashApp():
    return plot()

# メニューを表示
@app.route('/')
def nemu():
    return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect('/error')
        file = request.files['file']
        if file.filename == '':
            return redirect('/error')
        if file and allowed_file(file.filename):
            file_name = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            tr = train.TrainningExecution(file_path)
            tr.execute()
            return redirect("/plot/")
        else:
            return redirect('/error')

    return redirect("/")

@app.route('/download/iris', methods=['GET'])
def download_iris():
    downloadFileName = 'sample_iris.csv'
    downloadFile = './sample_files/sample_iris.csv'

    return send_file(downloadFile, as_attachment = True, \
        attachment_filename = downloadFileName, \
        mimetype = 'text/plain')

@app.route('/download/wine', methods=['GET'])
def download_wine():
    downloadFileName = 'sample_wine.csv'
    downloadFile = './sample_files/sample_wine.csv'

    return send_file(downloadFile, as_attachment = True, \
        attachment_filename = downloadFileName, \
        mimetype = 'text/plain')

@app.route('/error')
def error():
    return render_template("error.html")

# @app.errorhandler(Exception)
# def exception_handler(e):
#     return redirect('/error')

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True, port=9999)
