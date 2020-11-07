"""
This file is the entry point of the flask server. It initialize all configuration needed by other submodules, create server app.
For development server will be run on main at here.
"""
from flask import Flask, request
import nlp
import logging
from config import config
import exceptions


# Configuration
logging.basicConfig(filename=config['Log'], format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Load NLP modules.
nlp.Word2VecModelUser.load()
nlp.SummaryModelUser.load()

########################################################################################################################################
# Routing starts here.
########################################################################################################################################
app = Flask(__name__, static_folder=config['Build'], static_url_path='/')


@app.route('/')
def root():
    """ Redirects to index.html """
    return app.send_static_file('index.html')


########################################################################################################################################
# API Handler
########################################################################################################################################
@app.route('/api/nearby?words', method=['GET'])
def query_nearby_words():
    """ Handle nearby request. Provide neighbour words, score, and vector representation.
        Args:
            query: string of a word for the query.
            dim: dimension of the coordinates to be returns. 
        Returns:
            return json of words, score, and vector representation. 
    """
    # TODO: use nlp.Word2VecModelUser.sematic_field function. 
    # TODO: make sure to handle case when the word is not in vocab. try catch.
    return {}


@app.route('/api/summarize?topic')
def summarize_article():
    """
        Handle summarzier request. Given a word, pull page from wikipedia and summarizes the article.
        Args:
            topic: a topic for wikipedia page.
        Returns:
            summarized article from wikipedia.
    """
    # TODO: use wikipedia library for pulling page, and use nlp.SummaryModeluser.summarize function. 
    # TODO: make sure that the str passed into summarize function is  '/^[a-zA-Z0-9,.!? ]*$/'
    pass


########################################################################################################################################
# Error Handler & Error Page.
########################################################################################################################################
@app.errorhandler(exceptions.BadArgumentRequest)
def handle_bad_argument(error):
    return {
        'type': 'BadArgumentRequest',
        'msg': error.msg,
    }, error.code

    
########################################################################################################################################
# Run Dev Server
########################################################################################################################################
if __name__ == '__main__':
    logger.info('Development Server Started')
    print("Running dev server on localhost:8888")
    app.run(host='0.0.0.0', port=8888)