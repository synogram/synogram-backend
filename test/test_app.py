"""
    This file test the backend server and api.
"""
import pytest
from app.app import app

@pytest.fixture(scope="module")
def client():
    client = app.test_client()
    ctx = app.app_context()
    ctx.push()    
    yield client
    ctx.pop()

def test_route(client):
    response = client.get('/')
    assert response.status_code == 200

def test_query_nearby_words(client):
    pass

def test_summarize_article(client):
    pass


# TODO: add more testing.


