# Global Weather Prediction Using Machine Learning

## Overview
This project uses historical global temperature data to predict weather patterns using Random Forest Regression. The model analyzes land average temperatures, maximum temperatures, minimum temperatures, and land-ocean average temperatures to make predictions.

## Features
- Data preprocessing and cleaning
- Temperature conversion (Celsius to Fahrenheit)
- Correlation analysis with heatmap visualization
- Random Forest Regression model
- Model accuracy evaluation
- Feature importance analysis

## Prerequisites
- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - seaborn
  - matplotlib

## Dataset
The project uses the GlobalTemperatures dataset which includes:
- Land Average Temperature
- Land Maximum Temperature
- Land Minimum Temperature
- Land and Ocean Average Temperature

## Installation
1. Clone the repository
```bash
git clone https://github.com/Nizar04/Weather-Predicition-Model.git
cd weather-prediction-ml
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements-weather.txt
```

## Usage
1. Place your 'GlobalTemperatures.csv' in the project directory
2. Run the script:
```bash
python weather_prediction.py
```

## Model Performance
- Baseline Mean Absolute Error as benchmark
- Random Forest model accuracy: ~99%
- Feature importance visualization

## Future Improvements
- Add more weather features (humidity, pressure)
- Implement different ML algorithms for comparison
- Create API for real-time predictions
- Add cross-validation
- Extend prediction window

## License
MIT
