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
- [Project structure](#project-structure)
- [Getting Started](#getting_started)
- [Deployment](#deployment)
- [Usage](#usage)
- [Built Using](#built_using)
- [TODO](../TODO.md)
- [Contributing](../CONTRIBUTING.md)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

The fitness tracker leverage the power of machine learning techniques to classify various excercises the users is performing at various point in time.The core functionality of our model lies in its ability to recognize and categorize various exercises, including bench press, deadlift, row, squat, overhead press. This comprehensive exercise repertoire ensures that users can track their entire workout routine effortlessly. 

## Project structure
=========================================================
*Artifacts:* This folder contains the ingested data and all the files generated from preprocesssing the data and the model files after training
**.github/workflows:** This contains the ci/cd pipeline yaml file for deployment in aws EC2 instance
**log:** contains the log files throughout the project
**report:** Contains all the plot during the experiments
**src:**  This directory containings two folders;
      **components:** 
                  **Data ingestion:** Downloads the data from source into local machine and unzips it
                  **Data transformation:** Merge the all the csv data files together and perform some preprocessing techniques


## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them.

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running.

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo.

## 🔧 Running the tests <a name = "tests"></a>

Explain how to run the automated tests for this system.

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## 🎈 Usage <a name="usage"></a>

Add notes about how to use the system.

## 🚀 Deployment <a name = "deployment"></a>

Add additional notes about how to deploy this on a live system.

## ⛏️ Built Using <a name = "built_using"></a>

- [MongoDB](https://www.mongodb.com/) - Database
- [Express](https://expressjs.com/) - Server Framework
- [VueJs](https://vuejs.org/) - Web Framework
- [NodeJs](https://nodejs.org/en/) - Server Environment

## ✍️ Authors <a name = "authors"></a>

- [@kylelobo](https://github.com/kylelobo) - Idea & Initial work

See also the list of [contributors](https://github.com/kylelobo/The-Documentation-Compendium/contributors) who participated in this project.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Hat tip to anyone whose code was used
- Inspiration
- References
