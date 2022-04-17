import os

from flask import (
    render_template,
    redirect,
    request,
    flash)
from werkzeug.utils import secure_filename

from app import app

UPLOAD_FOLDER = 'app/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def upload_file():
    # get the submitted form data
    title = request.form.get('title')
    error = False
    # check if the post request has the title
    if title == '':
        error = True
        flash('Please provide a title', 'error')
    # check if the post request has the cover file and user
    # selected an actual file
    if 'cover' not in request.files:
        error = True
        flash('Please provide a book cover', 'error')
    file = request.files['cover']
    if file.filename == '':
        error = True
        flash('Please provide a book cover', 'error')
    if error:
        return redirect('/')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash("Title and Cover Successfully Uploaded!", 'success')
    return render_template('index.html')
