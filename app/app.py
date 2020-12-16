"""
This file is the entry point of the flask server. It initialize all configuration needed by other submodules, create server app.
For development server will be run on main at here.
"""
from flask import Flask, request
from flask_cors import CORS, cross_origin
import nlp
import logging
from .config import config
from . import exceptions
import wikipediaapi
import json
import re
import random
import requests

# Configuration
logging.basicConfig(filename=config['Log'], format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
)

# Load NLP modules.
nlp.WordEmbedModelUser.load()
nlp.SummaryModelUser.load()

########################################################################################################################################
# Routing starts here.
########################################################################################################################################
app = Flask(__name__, static_folder=config['Build'], static_url_path='/')
cors = CORS(app)

def word_check(word):
    exp = re.compile('^[a-zA-Z0-9,.!?\s]+$')
    result = exp.match(word)
    if result:
        return True
    else:
        raise ValueError

def pull_wiki_page(query):
    page = wiki.page(query)
    if page.exists():
        return page.summary
    else:
        raise exceptions.WikiPageNotFound()


@app.route('/', methods=['GET'])
def root():
    """ Redirects to index.html """
    # return app.send_static_file('index.html')
    return "Hello Flask"


########################################################################################################################################
# API Handler
########################################################################################################################################
# TODO: Handle help.

@app.route('/api/query/<string:query>/<int:topn>', methods=['GET'])
def query_nearby_words(query, topn):
    try:
        word_check(query)
        words, scores = nlp.WordEmbedModelUser.most_similar(query, topn=topn)
        return_dict = { 'words': words, 'scores': scores }
        return json.dumps(return_dict)   
    except KeyError:
        return 'Query not found', 400
    except ValueError:
        return 'Query not accepted', 400
    except:
        return 'Unknown problem', 500

@app.route('/api/summary/<string:query>', methods=['GET'])
def summarize_article(query):
    try:
        word_check(query)
        txt = pull_wiki_page(query)
        txt = re.sub('^[a-zA-Z0-9,.!?\s]+$', ' ', txt)
        content = nlp.SummaryModelUser.summarize(txt)        
        return content
    except ValueError:
        return 'Query not accepted', 400
    except exceptions.WikiPageNotFound:
        return 'Wikipedia page not found', 400
    except Exception as e:
        return 'Unknown Error' + str(e), 500 

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