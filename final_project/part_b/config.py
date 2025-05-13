import pandas as pd
import datetime

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
    TESLA_EPS = pd.read_csv("final_project/part_b/data/tesla_eps.csv", parse_dates=["publish_date"])
    NASDAQ_TICKER = "^IXIC"
    SP500_TICKER = "^GSPC"
    SAMPLE_STOCKS_FOR_CSAD = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META"
    ]

    FIRST_HERDING_PERIOD = {
        "start": datetime.date(2025, 1, 27),
        "end": datetime.date(2025, 3, 3)
    }

    SECOND_HERDING_PERIOD = {
        "start": datetime.date(2024, 10, 7),
        "end": datetime.date(2024, 12, 9)
    }

    THIRD_HERDING_PERIOD = {
        "start": datetime.date(2022, 3, 28),
        "end": datetime.date(2022, 5, 16)
    }

    FIRST_NON_HERDING_PERIOD = {
        "start": datetime.date(2023, 6, 1),
        "end": datetime.date(2023, 12, 31)
    }

    # SECOND_NON_HERDING_PERIOD = {
    #     "start": datetime.date(2023, 6, 1),
    #     "end": datetime.date(2023, 12, 31)
    # }

    # THIRD_NON_HERDING_PERIOD = {
    #     "start": datetime.date(2023, 6, 1),
    #     "end": datetime.date(2023, 12, 31)
    # }