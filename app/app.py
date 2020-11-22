"""
This file is the entry point of the flask server. It initialize all configuration needed by other submodules, create server app.
For development server will be run on main at here.
"""
from flask import Flask, request
import nlp
import logging
from config import config
import exceptions
import wikipedia
import json
import re


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

def wordCheck(word):
    exp = re.compile('^[a-zA-Z0-9,.!?\s]+$')
    result = exp.match(word)
    if result:
        return True
    else:
        raise ValueError


@app.route('/', methods=['GET'])
def root():
    """ Redirects to index.html """
    # return app.send_static_file('index.html')
    return "Hello Flask"


########################################################################################################################################
# API Handler
########################################################################################################################################
@app.route('/api/query/<string:query>/<int:dim>', methods=['GET'])
def query_nearby_words(query, dim):
    try:
        wordCheck(query)
        returnValue = nlp.Word2VecModelUser.sematic_field(query, dim=dim, topn=10)
        returnDict = {word: {'score': score, 'vector': vector} for word, score, vector in returnValue}
        return json.dumps(returnDict)   
    except KeyError:
        return 'Query not found'
    except ValueError:
        return 'Query not accepted'
    except:
        return 'Other problem'
    # TODO: use nlp.Word2VecModelUser.sematic_field function. 
    # TODO: make sure to handle case when the word is not in vocab. try catch.

@app.route('/api/summary/<string:query>', methods=['GET'])
def summarize_article(query):
    try:
        wordCheck(query)
        page = wikipedia.page(query)
        content = nlp.SummaryModelUser.summarize(page.content)
        return page.content
    except ValueError:
        return 'Query not accepted'
    except:
        return "Could not find page"
    # TODO: use wikipedia library for pulling page, and use nlp.SummaryModeluser.summarize function. 
    # TODO: make sure that the str passed into summarize function is  '/^[a-zA-Z0-9,.!? ]*$/'


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
    print("Running dev server on localhost:5000")
    app.debug = True
    app.run(host='0.0.0.0', port=5000)