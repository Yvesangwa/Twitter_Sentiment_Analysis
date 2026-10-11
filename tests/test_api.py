import pytest
from app.app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


# Test 1: API is running
def test_home(client):
    response = client.get("/")

    assert response.status_code == 200

    data = response.get_json()

    assert data["message"] == (
        "Twitter Sentiment Analysis API is running"
    )


# Test 2: Positive sentiment
def test_positive_sentiment(client):
    response = client.post(
        "/predict",
        json={"tweet": "I absolutely love this amazing movie!"}
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["sentiment"] == "Positive"


# Test 3: Negative sentiment
def test_negative_sentiment(client):
    response = client.post(
        "/predict",
        json={"tweet": "This is the worst experience ever!"}
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["sentiment"] == "Negative"


# Test 4: Missing tweet
def test_missing_tweet(client):
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 400


# Test 5: Empty tweet
def test_empty_tweet(client):
    response = client.post(
        "/predict",
        json={"tweet": ""}
    )

    assert response.status_code == 400
