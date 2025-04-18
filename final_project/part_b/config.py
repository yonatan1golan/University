import pandas as pd

class CONFIG:
    WEEKDAYS_DICTONARY = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    QUARTERS_DICTONARY = {
        1: "Q1",
        2: "Q2",
        3: "Q3",
        4: "Q4",
    }

    TESLA_TICKER = "TSLA"
    TESLA_EPS = pd.read_csv("part_b/data/tesla_eps.csv", parse_dates=["publish_date"])
    NASDAQ_TICKER = "^IXIC"
    SP500_TICKER = "^GSPC"
    SAMPLE_STOCKS_FOR_CSAD = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META"
    ]
