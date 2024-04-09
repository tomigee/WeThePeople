# WeThePeople: Predicting Economic Impact of Legislation
## Project Overview:

WeThePeople is a machine learning project designed to predict the economic impact of US legislation. It utilizes a transformer-based model to analyze raw text data from the US legislative corpus and historical economic time series data.

## Technical Approach:

* **Data Acquisition:** Custom Python packages were developed to efficiently source data from the [US Congress API](https://github.com/tomigee/Congress) and the [FRED API](https://github.com/tomigee/fred).
* **Data Preprocessing:** A data pipeline automates data sourcing, cleaning, and organization, ensuring high-quality data for model training.
* **Machine Learning Model:** A transformer-based NLP model utilizes the pre-trained BERT architecture to analyze legislative text and a custom decoder to ingest historical economic data. This model forecasts economic trends post-legislation.
* **Evaluation:** The model achieves high accuracy, demonstrating less than 10% error on econometric time-series predictions.

## Project Significance:

WeThePeople provides a valuable tool for:

* Analyzing the potential economic consequences of proposed legislation.
* Informing economic policy decisions.
* Strengthening the understanding of the relationship between legislation and economic outcomes.

## Future Work:

* Enhance the model's ability to handle complex and nuanced legislative language.
* Strengthen the model's ability to discern seasonality in time-series data.
* Integrate additional contemporaneous data sources to further enrich the model's insights.
* Develop a user-friendly interface for interactive exploration of predicted economic impacts.
* Expand data pipeline support for text in PDF file format
