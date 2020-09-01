# Building the collaborative-filtering recommendation system using yelp dataset

## Project description
This project is a mini-project in class Data Mining. The goal is to be familiar with MinHash, Locality Sensitive Hashing, and different types of collaborative filtering recommendation system.

## Programming language and libraries
Python, Spark, scala, only use standard Python libraries and Spark RDD

## Procedure
- Identified similar businesses by implementing the Locality Sensitive Hashing algorithm with Jaccard similarity.
- Built the User-based CF recommendation system using training data(22.9M), the RMSE was 1.0756, the processing time was 190s.
- Built the item-based CF recommendation system using the same training data. The RMSE was 1.0754, the running time was 657s.

## Data
The dataset is a subset from the Yelp dataset
