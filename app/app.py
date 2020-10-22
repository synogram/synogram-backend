"""
This file is the entry point of the flask server. It initialize all configuration needed by other submodules, create server app.
For development server will be run on main at here.
"""
from flask import Flask, request
import nlp.models as nlp
import logging
from config import config
import exceptions


# Configuration
logging.basicConfig(filename=config['Log'], format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

nlp.init()

########################################################################################################################################
# Routing starts here.
########################################################################################################################################
app = Flask(__name__, static_folder=config['Build'], static_url_path='/')


@app.route('/')
def root():
    """ Redirects to index.html """
    return app.send_static_file('index.html')


@app.route('/api/web', methods=['GET'])
def query_web():
    """ Returns Web based on query
        Args:
            query: string of a word for the query.
            dim: dimension of the coordinates to be returns. 
        Returns:
            return json(dictionary) of the web.
    """
    if request.method == 'GET':
        query = request.args.get('query')
        dim = int(request.args.get('dim'))
        return nlp.wiki2vec.build_web(query, dim), 200
    else:
        return 'bad request', 400


@app.errorhandler(exceptions.BadArgumentRequest)
def handle_bad_argument(error):
    return {
        'type': 'BadArgumentRequest',
        'msg': error.msg,
    }, error.code

    

if __name__ == '__main__':
    logger.info('Development Server Started')
    print("Running dev server on localhost:8888")
    app.run(host='0.0.0.0', port=8888)