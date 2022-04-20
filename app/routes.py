import json
import os

from flask import (
    render_template,
    redirect,
    request,
    flash)
from werkzeug.utils import secure_filename

from app import app

from app.ml.CountPattern import cosine_similarity_func

UPLOAD_FOLDER = 'app/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def combine_results_with_proper_keys(result_dict):
    """
    Swap the integer keys for the actual categories so that the
    data cane be easily used in Chart.js visualizations
    :param result_dict: dictionary of results with integer keys
    :return: dictionary of results with the Categories as keys
    """
    merged = {}

    categories = {1: "Graphic Novels Anime-Manga, and Comics",
                  2: "Transport, Travel, and Sport", 4: "Food and Drink", 5: "Home, Hobbies, and Crafts",
                  6: "Computing and Video Games", 7: "Religion",
                  8: "Literature, Poetry, and Plays", 9: "Humor", 10: "Language and Reference", 11: "Romance",
                  12: "Biography", 13: "History",
                  14: "Teen and Young Adult", 15: "Sci-Fi and Fantasy", 16: "Children",
                  17: "Science, Psychology, and Self Help",
                  18: "Crime, Mystery, and Thriller"}

    for key, name in categories.items():
        merged[name] = result_dict[key]

    return merged


# the routes aren't too complex now, but we
# probably should set up a Controller if
# they get any bigger
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/results', methods=['POST', 'GET'])
def upload_file():
    # the returned result and categories list
    result = {}
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
        flash("Here are the results!", 'success')
        result = cosine_similarity_func(title)
    return render_template('results.html', result=json.dumps(combine_results_with_proper_keys(result)))
