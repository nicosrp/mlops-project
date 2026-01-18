# Machine Learning Operations - Final Project

## Project Description:

As football enthusiasts, this project aims to shift the perspective from simply being fans to using match data to gain insights into what drives a team’s success. We aim to predict the probability of each match outcome (home win, draw, or away win) for each team and each match. We are using the [Football Match Probability Prediction](https://www.kaggle.com/competitions/football-match-probability-prediction/data) dataset from the Kaggle competition. The dataset is composed of more than 150,000 football matches in the world, with descriptive features on home vs. away teams, league, and coaches, in addition to historical information on the previous 10 matches for both home and away teams, such as date, number of goals scored, number of goals conceded, and more.

A potential pipeline could include training a multiclass probabilistic classifier with cross-entropy loss, tracking experiments and model versions using Weights and Biases, and deploying an ML service that predicts home, draw, and away win probabilities for incoming match data. Results could be benchmarked against dummy-like equal probability classifications or the previous winners' choice. We may also explore an LSTM due to the sequence-nature of the data.

## Project structure

The directory structure is as follows: 
```txt
├── .github/                        # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── lint-test-build.yaml
├── configs/                        # Configuration files
├── data/                           # Data directory
│   ├── train.csv
│   └── test.csv
|   └── train_target_and_scores.csv
├── dockerfiles/                    # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                           # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                         # Trained models
├── notebooks/                
├── reports/                        # Reports
│   └── figures/
├── src/                            # Source code
│   ├── mlops_project/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                          
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml                  # Python project file
├── README.md                       # Project README
├── requirements.txt                # Project dependencies
├── requirements_dev.txt            # Development requirements
└── tasks.py                        # Project tasks