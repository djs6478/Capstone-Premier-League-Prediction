# Capstone-Premier-League-Prediction

## Overview

The sport of soccer and sports gambling are growing in popularity. One of the most popular forms of soccer gambling is table prediction gambling. In table prediction gambling, people gamble on where they think teams will finish in the table at the end of the year. The Premier League is one of the most popular leagues in the world and is the league I will be using.

I will create a simiulation model to predict the probablities of where teams will finish in the table. Using these probabilties, the stakeholder will be able to create their gambling odds. 

## Stakeholder 

Through this project we'd like to with Draft Kings. They're one of the premier sportsbooks in America. As online gambling contiues to become legalized, Draft Kings will be on the forefront of the market. 

## Business and Data Understanding

I gathered my data from SofaScore.com. The data was gathered from the years 2009 to 2019. The original data was in txt form. The first thing I did was convert the data into csv files. The original data itself contained 8 values. These were:

1. Date

2. Weekday

3. HomeTeam

4. HomeScore

5. AwayScore

6. AwayTeam

7. Winner

8. Stadium

Using this cleaned data I then created Team Data which contains 27 unique values. 



## Modeling

We iterated over several different types of models. For our initital train test split, we created a holdout set of 10% to test on once we came to a final model, then we did another train test split that was a 75/25 split. 

For our first simple model, we chose a Decision Tree because with a Decision Tree we wouldn't have to scale any data. We concluded that our Decision Tree was overfit. It scored 100% accuracy on the training data and about 73% accuracy on the test data. 

We then tried a Random Forest Classifier, Logistic Regression, XGBoost, and a KNeighbors Classifier. We created a Pipeline so we could OneHotEncode the columns more rapidly. We also performed a Grid Search that allowed us to tune the hyperparameters to get a more accurate model. 

Our final model was an XGBoost model that gave us the best results on precision without overfitting. 

## Evaluation

The final XGBoost Model we came to was slightly overfit on the training data but performed well on our unseen holdout data. Out of 5438 wells that were classified as functional only 1180 of them were actually non functional.  A precision score of 78% means that we have a pretty low false positive rate. Test precision score is 4 points higher than our original simple decision tree model, but that model was severely overfit due to there being no max depth limit in that model. 

## Conclusion

We reccommend that Wells of Life use this model in conjuction with their own resources to identify faulty wells. Our model will still classify some non -functional wells as functional 20% of the time so further investigation will be needed at times.

## Future Work

We'd like to further engineer more features to give our model better data to make predictions off of. We'd also like to further tune hyperparameters to optimize our model's performance on unseen data. With more time we would like to input climate data into our model to see how climate may be affecting wells, as well. 

Overall, we'd like to continuously gather more information on the wells in Tanzania to add to our data and improve our model.

## Presentation Link 

[Presentation](https://github.com/djs6478/Well-Project/blob/main/presentation.pdf)

## Repository Navigation

```
├── README.md                    <- The top-level README for reviewers of this project. 
├── data                         <- Sourced externally and generated from code. 
├── notebooks                    <- Folder containing Jen, Wayne, and Derek Jupyter Notebooks housing individual work for this project. 
├── EDA                          <- Exploratory Data Analysis containing Jen, Wayne, and Derek EDA Jupyter Notebooks. 
├── .gitignore                   <- Plain text file where each line contains a pattern for files/directories to ignore.
├── final_notebook.ipynb         <- Final Jupyter notebook for this project, containing a final XGBoost model that tested on the holdout set. 
└── presentation.pdf             <- PDF of the presentation slides for this project.                 