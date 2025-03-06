# Stock Signal 1.0

This project is a stock screening tool that fetches stock data, calculates various technical indicators, scores the stocks based on these indicators, and sends an email report of the top recommended stocks.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Technical Indicators](#technical-indicators)
- [Email Notification](#email-notification)
- [Scheduled Task](#scheduled-task)
- [Logging](#logging)

## Requirements

1. Python 3.10

## Installation

1. Clone the repository:
    ```sh
    git clone git@github.com:weiluo8791/stock-signal.git
    cd stock-signal
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv stock_signal_310
    source stock_signal_310/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

Update the [EMAIL_CONFIG](http://_vscodecontentref_/1) and [INDICATOR_WEIGHTS](http://_vscodecontentref_/2) dictionaries in [stock-signal.py](http://_vscodecontentref_/3) to match your preferences.

## Usage

To run the stock screening tool, execute the following command:
```sh
python stock-signal.py
```

## Technical Indicators
The tool calculates the following technical indicators:

- Bollinger Bands
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Ichimoku Cloud
- Bias
- ADX (Average Directional Index)
- Parabolic SAR
- OBV (On-Balance Volume)
- Moving Averages (5-day, 20-day, 60-day)

## Email Notification
The tool sends an email report of the top 10 recommended stocks based on the calculated scores. The email configuration is defined in the EMAIL_CONFIG dictionary.

## Scheduled Task
The tool can be scheduled to run at regular intervals using the schedule library. The interval is defined in the INDICATOR_WEIGHTS['run_interval'] parameter.

## Logging
The tool logs its activities to both a log file (stock_scanner.log) and the console. The logging configuration is defined in the logging.basicConfig function.

## License
This project is licensed under the MIT License. See the LICENSE file for details.