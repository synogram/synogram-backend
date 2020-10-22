# Latest python gives error with scikitlearn 
FROM python:3.7.4

# Install nodejs and yarn
# RUN apt-get update
# RUN apt-get -y install nodejs
# RUN apt-get -y install npm 
# RUN npm install npm@latest -g
# RUN npm install -g yarn

COPY . /app
WORKDIR /app

RUN pip install -U pip
RUN pip install -r requirements.txt

# Remove this line if don't want to build.
# RUN cd app/client && yarn install && cd .. && cd ..
# RUN cd app/client && yarn build && cd .. && cd ..

ENTRYPOINT ["python"]
CMD ["app/app.py"]