<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTnrdgav0Nvkm5M8rTIdupyKpyOd_qUNwHfug&usqp=CAU" alt="Project logo"></a>
</p>

<h3 align="center">EXERCISE FITNESS ML TRACKER</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> This fitness tracker is designed to classify different exercises performed during a workout session using machine learning techniques. The tracker can identify the following exercises: bench press, deadlift, row,  squat, overhead press,and also detect rest periods when the user is not performing any exercise.
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Data](#data)
- [Project structure](#project-structure)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)


## 🧐 About <a name = "about"></a>

The fitness tracker leverage the power of machine learning techniques to classify various excercises the users is performing at various point in time.The core functionality of our model lies in its ability to recognize and categorize various exercises, including bench press, deadlift, row, squat, overhead press. This comprehensive exercise repertoire ensures that users can track their entire workout routine effortlessly. 

video link of the project : https://x.com/oboro_gideon/status/1685775882269368320?t=ZV1XoVVWB0S8t33UANW5KQ&s=19

## Data
The model was trained with both accelerometer and gyroscope data collected over a period of time with five different participants performing  all the various excercises both medium and heavy sets using the sensors in a fitness watch.

### gyrocsope data:
- epoch (ms) : unix time
- time (01:00) : datetime
- elapsed (s) : duration
- x-axis (deg/s) : horizontal axis
- y-axis (deg/s) : vertical axis
- z-axis (deg/s) : depth

### accelerometer data:
- epoch (ms) : unix time
- time (01:00) : datetime
- elapsed (s) : duration
- x-axis (g) : horizontal axis 
- y-axis (g) : vertical axis
- z-axis (g) :depth

## Project structure
- Artifacts
- .github/workflows
- log
- report 
- src
  - components
    - Data ingestion
    - Data transformation
    - outlier removal 
    - build features
    - data modeling
  - pipeline
    - training pipeline
    - predict pipeline
  - visualization/experiments
    - visualize
    - detect outliers
    - build features
    - model train
  - exception
  - logger
- venv
- .gitignore
- app
- Dockerfile
- dvc.lock
- dvc.yaml
- model_metrics.json
- params.yaml
- requirements
- setup
- utils


## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing

A step by step series of examples that tell you how to get a development env running.

* __Clone the repository:__ Run this in your terminal

```
git clone git@github.com:GIDDY269/fitness_tracker_ml.git
```

* __create a vitrual environment:__

```
conda create -p venv python==3.9 -y
```

* __install the dependencies:__
```
pip install -r requirements.txt
```

## 🎈 Usage <a name="usage"></a>

To use this service , open your terminal and run

```
uvicorn app:app --reload
```
Then copy the url and paste in your web browser, add `/docs` at the end to open a fastapi swagger ui 


## ⛏️ Built Using <a name = "built_using"></a>

- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [Scikit learn](https://scikit-learn.org/stable/) - machine learning
- [Docker](https://www.docker.com/) - building container
- [DVC](https://dvc.org/) - data versioning


