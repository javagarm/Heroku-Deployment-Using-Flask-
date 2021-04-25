from flask import Flask, request, render_template, jsonify
import yfinance as yf
import numpy as np
import pickle
import pandas as pd
import datetime
from flask import Flask, send_file
from forex_python.converter import CurrencyRates
import datetime
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

import seaborn
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# In[ ]:


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [pd.to_datetime(x).value for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    
    symbol='GC=F'
    ticker = yf.Ticker(symbol)
    if date.today().weekday()== 5 or 6:
        todays_data = ticker.history(period='2d')
    else:
        todays_data = ticker.history(period='1d')
    output1=todays_data['Close'][0]
    output1 = round(output1,4)
    


    c = CurrencyRates()
    today_date=datetime.today().strftime('%d/%m/%Y')

    dt = pd.to_datetime(today_date)

    output_date=(c.get_rate('USD', 'INR', dt))
    dt1 = output_date*output
    dt1=round(dt1,2)

    return render_template('homepage.html', cur_rate=' USD : {}'.format(output1),prediction_text='The Gold Rate predicted is $ {}'.format(output),dt_inr = 'INR : {}'.format(dt1))
    


# In[ ]:


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict(pd.to_datetime(data))
    

    output = prediction[0]
    return jsonify(output)


# API Route for pulling the stock quote
@app.route("/quote")
def display_quote():
	# get a stock ticker symbol from the query string
	# default to AAPL
	symbol = request.args.get('symbol', default="GC=F")

	# pull the stock quote
	quote = yf.Ticker(symbol)

	#return the object via the HTTP Response
	return jsonify(quote.info)

# API route for pulling the stock history
@app.route("/history")
def display_history():
	#get the query string parameters
	symbol = request.args.get('symbol', default="GC=F")
	period = request.args.get('period', default="1y")
	interval = request.args.get('interval', default="1mo")

	#pull the quote
	quote = yf.Ticker(symbol)	
	#use the quote to pull the historical data from Yahoo finance
	hist = quote.history(period=period, interval=interval)
	#convert the historical data to JSON
	data = hist.to_json()
	#return the JSON in the HTTP response
	return data

# This is the / route, or the main landing page route.
@app.route("/")
def home():
	# we will use Flask's render_template method to render a website template.
    return render_template("homepage.html")

# run the flask app.
if __name__ == "__main__":
	app.run(debug=True)
