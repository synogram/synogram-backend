from allennlp.predictors.predictor import Predictor

model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
predictor = Predictor.from_path(model_url)  # load the model

text = "Eva and Martha didn't want their friend Jenny to feel lonely so they invited her to the party."
prediction = predictor.predict(document=text)  # get prediction

print(prediction['clusters'])  # list of clusters (the indices of spaCy tokens)
print(predictor.coref_resolved(text))  # resolved text