# FIN-407 - Project


This project investigates the potential for predicting the financial returns of stocks within the agro-industrial sector using machine learning techniques. The agro industry is influenced by a wide range of external and internal factors, and our goal is to capture this complexity through a multi-faceted modeling approach.

We focus on three categories of predictive variables:
- **Meteorological data**, which reflect environmental conditions that directly affect agricultural output;
- **Sentiment analysis derived from industry-related news**, capturing the market's perception and expectations;
- **Firm-level characteristics turned into tradable long–short portfolios**, representing cross-sectional financial signals that capture a range of economic themes such as value, momentum, profitability, and investment behavior.

By integrating these three dimensions, we aim to build predictive models that are not only data-rich but also context-aware.

The project report can be found under Machine Learning Predicting Agri-Food Industry returns pdf

## Project Structure

```
FIN-407-Project/
│
├── FIN-407_Report.pdf
├── BackTesting
│   └── strategies (1).py
├── Model
│   ├── All Features
│   │   ├── Results
│   │   │   ├── predictions.csv
│   │   │   └── training_losses.csv
│   │   └── Weights & Parameters
│   │       ├── best_hyperparameters.json
│   │       ├── training_dataset.pth
│   │       └── weights_model.pth
│   ├── Loss_graphs.py
│   ├── Restricted Features
│   │   ├── Results
│   │   │   ├── predictions.csv
│   │   │   └── training_losses.csv
│   │   └── Weights & Parameters
│   │       ├── best_hyperparameters.json
│   │       ├── training_dataset.pth
│   │       └── weights_model.pth
│   ├── TFT.py
│   ├── create_dataloader.py
│   ├── hyperpar_opt.py
│   ├── main.py
│   └── predict_returns.py
├── README.md
└── data
    ├── Data Files
    │   ├── final_data_non-normalized.csv
    │   ├── final_data_reduced_non-normalized.csv
    │   ├── raw data weather
    │   │   ├── illinois_weather_data.csv
    │   │   ├── indiana_weather_data.csv
    │   │   ├── iowa_weather_data.csv
    │   │   ├── kansas_weather_data.csv
    │   │   ├── minnesota_weather_data.csv
    │   │   ├── nebraska_weather_data.csv
    │   │   └── sodakota_weather_data.csv
    │   ├── risk-free.csv
    │   ├── sentiment_data.csv
    │   └── weather_data_00_24.csv
    ├── Data Processing
    │   ├── JKP_factors.ipynb
    │   ├── News Dataset.ipynb
    │   ├── dailyWorker.py
    │   ├── extract_weather.ipynb
    │   ├── extraction.py
    │   ├── firms_characteristics.ipynb
    │   ├── firms_characteristics_data_download.py
    │   ├── reg_features.ipynb
    │   └── target_returns_and_final_data.ipynb
    └── Sentiment Analysis
        └── Sentiment_analysis.ipynb
```
## Notes

The model was trained twice (once using all available features, and once using only the most important features identified through regression analysis). However, the training code is provided only once, as the only differences between the two runs were the input data file name and the list of time-varying columns.

---

**Contributors**: Christopher Soriano, Candice Busson, Marine De Rocquigny, Timothé Dard, and Cyprien Tordo
