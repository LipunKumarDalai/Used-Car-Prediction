FROM python:3.11.3-slim

WORKDIR /App

COPY . /App

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV STREAMLIT_APP=src/App.py

CMD ["streamlit", "run", "src/App.py", "--server.port=5000", "--server.address=0.0.0.0"]
