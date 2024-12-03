## End to End ML Project

# Employee Retention Prediction(Classification)

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Run](#run)
  * [Deployement on render](#deployement-on-render)
  * [Directory Tree](#directory-tree)
  * [To Do](#to-do)
  * [Bug / Feature Request](#bug---feature-request)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [License](#license)
  * [Credits](#credits)


## Demo :
Now no demo is given <br>
<!-- Link: [Demo project](https://dog-vs-cat-clf.onrender.com) -->



<!-- <!-- [![](https://imgur.com/s4FWb9b)](https://ipcc.rohitswami.com) -->
## Employee Leave or Not Classifiers:
![Dog vs Cat](https://www.questionpro.com/blog/wp-content/uploads/2023/04/employee-retention.jpg)

## Overview
This project focuses on the classification of employee data to predict if an employee will take leave or not based on various features, such as educational background, work experience, demographics, and more. The task uses machine learning to analyze various employee-related factors and predict their leave behavior.

This repository contains code for training multiple machine learning classifiers on a given dataset to predict employees. The models are implemented using the scikit-learn library and include a wide range of supervised learning algorithms suitable for classification tasks.

# Model accuracy of 81%
The following classifiers are applied to the dataset:<br>

-Logistic Regression<br>
-K-Nearest Neighbors (KNN)<br>
-Support Vector Machine (SVM)<br>
-Decision Tree<br>
-Random Forest<br>
-Gradient Boosting<br>
-Naive Bayes<br>
-Multi-layer Perceptron (MLP)<br>
-XGBoost<br>

The algorithms is MLP <br>
Best model 'MLP' saved with an accuracy of 0.81


## Motivation
Employee retention is a critical challenge for many companies. Predicting whether an employee is likely to stay or leave helps businesses make informed decisions on how to improve retention strategies and reduce turnover costs. This project uses machine learning classification algorithms to solve the employee retention problem effectively.

.Key Motivations:<br>
-Real-World Applications: Companies can use this model to predict employee turnover and take necessary actions to retain employees.<br>
-Understanding Machine Learning: By implementing multiple classifiers, this project serves as a comprehensive example of various machine learning techniques and their use in solving business problems.

## Technical Aspect

This project is divided into two major parts:

Training machine learning Models:

We train multiple machine learning algorithms on the Employee Retention dataset. All models are implemented using scikit-learn, a Python library for machine learning. Evaluation is performed using performance metrics such as accuracy, precision, recall, and F1-score.
<br>

Building and Hosting a Flask Web App on Render:

A Flask web application is built to interact with the trained models and make real-time predictions based on user input.
The app is deployed using Render to provide easy access via the web.
Users can submit their data via a simple web interface and receive Employee Retention predictions
    - 

## Installation
The Code is written in Python 3.10. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

# To clone the repository

```bash

gh repo clone Creator-Turbo/Employee-Retention-Prediction-Deployment

```
# Install dependencies: (all lib)
```bash
pip install -r requirements.txt
```



## Run
To train the ML leaning models:
 To run the Flask web app locally
```bash
python app.py

```
# Deployment on Render

## To deploy the Flask web app on Render:
Push your code to GitHub.<br>
Go to Render and create a new web service.<br>
Connect your GitHub repository to Render.<br>
Set up the environment variables if required (e.g., API keys, database credentials).<br>
Deploy and your app will be live!



## Directory Tree 
```
.
employee_classification/
├── .gitignore
├── best_model_MLP.pkl
├── README.md
├── requirements.txt

```

## To Do




## Bug / Feature Request
If you encounter any bugs or want to request a new feature, please open an issue on GitHub. We welcome contributions!




## Technologies Used
Python 3.10<br> 
scikit-learn<br>
TensorFlow <br>
Flask (for web app development)  <br>
Render (for hosting and deployment)  <br>
pandas (for data manipulation) <br>
numpy (for numerical operations)  <br>
matplotlib (for visualizations) <br>



![](https://forthebadge.com/images/badges/made-with-python.svg)


[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/260px-Scikit_learn_logo_small.svg.png" width=170>](https://pandas.pydata.org/docs/)
[<img target="_blank" src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*RWkQ0Fziw792xa0S" width=170>](https://pandas.pydata.org/docs/)
  [<img target="_blank" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSDzf1RMK1iHKjAswDiqbFB8f3by6mLO89eir-Q4LJioPuq9yOrhvpw2d3Ms1u8NLlzsMQ&usqp=CAU" width=280>](https://matplotlib.org/stable/index.html) 
 [<img target="_blank" src="https://icon2.cleanpng.com/20180829/okc/kisspng-flask-python-web-framework-representational-state-flask-stickker-1713946755581.webp" width=170>](https://flask.palletsprojects.com/en/stable/) 
 [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/512px-NumPy_logo_2020.svg.png" width=200>](https://aws.amazon.com/s3/) 






## Team
This project was developed by:

Bablu kumar pandey

<!-- Collaborator Name -->




## Credits

Special thanks to the contributors of the scikit-learn library for their fantastic machine learning tools.

