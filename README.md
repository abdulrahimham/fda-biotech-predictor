# FDA Biotech Stock Predictor

A machine learning project that predicts whether a drug company's stock 
will go up or down after the US government approves or rejects their drug.

## The Problem

When the US government (the FDA) decides whether a new drug is safe enough 
to sell, it is huge news for the company that made it. If the drug gets 
approved, the stock can shoot up 20 to 30% in a day. If it gets rejected, 
it can crash just as fast.

This project builds a model that looks at publicly available data and 
predicts which direction the stock will move in the 72 hours after 
the decision.

## Results

| What we measured | Result |
|-----------------|--------|
| Did the model predict UP vs DOWN correctly? | 66.7% of the time |
| What would random guessing get? | 50.0% |
| How much better is the model than guessing? | +16.7% |
| How far off were the magnitude predictions? | 4.08 percentage points |

## Where the Data Comes From

No Kaggle. No paid datasets. Everything is free and publicly available.

| Source | What it is | What I used it for |
|--------|-----------|-------------------|
| OpenFDA API | The FDA's official database of drug decisions | List of every branded drug approval since 2010 |
| SEC EDGAR | Public database of company announcements | Text companies published around each approval |
| ClinicalTrials.gov | Government database of medical studies | Trial size and phase for each drug |
| Yahoo Finance | Historical stock prices | How much each stock moved after the decision |

## How It Works

**Step 1: Collect the data**

The code automatically downloads drug approval decisions from the FDA, 
matches each one to a publicly traded company, and pulls the stock 
price history around that date.

**Step 2: Merge everything together**

Four separate datasets get joined into one master table using the drug 
application number as the common link — like stapling four spreadsheets 
together using a shared ID.

**Step 3: Engineer features**

Raw data gets transformed into numbers the model can learn from. For 
example, trial enrollment numbers get log scaled so a trial with 10,000 
patients does not completely overpower one with 100.

**Step 4: Train the model**

A Random Forest model learns patterns from historical approval events. 
Things like how big the trial was, what phase it was in, and how volatile 
the stock already was all contribute to the prediction.

**Step 5: Evaluate honestly**

The model is tested on events from 2023 onwards that it never saw during 
training. This simulates how it would perform in the real world.

**Step 6: Show the results**

A Streamlit web app displays everything interactively including the 
predictions, the feature importance, and how the model performed on 
each event.

## Why I Made Certain Choices

**Why only branded drugs and not generics?**
Generic drugs are cheap copies of existing drugs. Their approvals barely 
move stock prices because there is nothing surprising about them. I focused 
only on new branded drugs where the FDA decision is a major event.

**Why split the data by time instead of randomly?**
If I randomly split the data, the model could train on a 2024 event and 
test on a 2015 event. That would mean the model secretly saw the future 
during training, which is cheating. I trained on 2010 to 2022 and tested 
on 2023 onwards.

**Why Random Forest instead of a neural network?**
With 65 events, a neural network would just memorize the training data 
and fail completely on new data. Random Forest works well on small datasets 
and is easy to interpret and explain.

## Tech Stack

Python and pandas for data wrangling, scikit-learn for the model, 
Streamlit and Plotly for the dashboard, and public REST APIs for 
all data collection.

## Setup

```bash
git clone https://github.com/abdulrahimham/fda-biotech-predictor
cd fda-biotech-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 src/ingest/fda_ingest.py
python3 src/ingest/stock_ingest.py
python3 src/ingest/sec_ingest.py
python3 src/ingest/clinical_trials_ingest.py
python3 src/processing/merge_datasets.py
python3 src/features/feature_engineering.py
python3 src/models/train_model.py
streamlit run src/dashboard/dashboard.py
```

## Project 
da-biotech-predictor/
├── src/
│   ├── ingest/          # Data collection from each source
│   ├── processing/      # Merging all datasets together
│   ├── features/        # Feature engineering and scaling
│   ├── models/          # Model training and evaluation
│   └── dashboard/       # Streamlit web app
├── data/
│   ├── raw/             # Downloaded source data
│   └── processed/       # Cleaned and model ready data
└── requirements.txt

Built as a portfolio project by a Data Science student at UCSD.