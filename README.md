# MDM3_Code4Thought

## Model GUI

This repository contains a Python GUI built using Dash and Plotly that allows you to customize and compare four different time series models. The GUI allows you to choose the model type (short-term ARIMA, short-term random walk, long-term ARIMA, or long-term random walk) and the number of change points to detect.

The code for the GUI is located in the `model_gui` folder, and the main script to run the GUI is called `main.py`. To run the GUI, simply navigate to the `model_gui` folder and run the `main.py` script:

```
cd model_gui
python main.py
```

This will start the GUI on your local machine, and you can access it by opening a web browser and navigating to `http://localhost:8050`.

Please note that you will need to install the necessary libraries (Dash, Plotly, pandas, numpy, and statsmodels) in order to run the GUI. You can do this using the following command:
```
pip install dash plotly pandas numpy statsmodels
```
