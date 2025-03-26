# newCarsPython

You can use this project to ask questions about a specific resource that you have. 

cars.py is for reddit

scrapeFromWebsite2.py is for a website

scrapeLocal.py is for a local resource 

Regardless, all of them will be turned to chunks, vectorized, and inserted in lanceDB

chatbot is ran on streamlit


create .env file
client_id= "get client id from reddit"
client_secret="get client secret from reddit"
user_agent="get use agent from reddit"
subreddit="this is a real subreddit. It must be the exact name."
table="This can be anything you like. It is simply the name of the table in lancedb"
API_KEY = "your api key from deepseek"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

Steps to run
1. create a .env file
2. go to the repository in command line
3. run "python3.10 -m venv venv"
4. run "source venv/bin/activate"
5. run "pip install -r requirements.txt"
6. run cars.py/scrapeFromWebsite2.py/scrapeFromWebsite2.py
7. run "streamlit run chat.py"




