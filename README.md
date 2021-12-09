# Capstone-Premier-League-Prediction

## Overview

The sport of soccer and sports gambling are growing in popularity. One of the most popular forms of soccer gambling is table prediction gambling. In table prediction gambling, people gamble on where they think teams will finish in the table at the end of the year. The Premier League is one of the most popular leagues in the world and is the league I will be using.

I will create a simiulation model to predict the probablities of where teams will finish in the table. Using these probabilties, the stakeholder will be able to create their gambling odds. 

## Stakeholder 

Through this project we'd like to with Draft Kings. They're one of the premier sportsbooks in America. As online gambling contiues to become legalized, Draft Kings will be on the forefront of the market. 

## Business and Data Understanding

I gathered my data from SofaScore.com. The data was gathered from the years 2009 to 2019. The original data was in txt form. The first thing I did was convert the raw data into csv files which is titled clean data. The data contains 8 columns. These were the date, weekday, HomeTeam, HomeScore, AwayScore, AwayTeam, Winner and Stadium.
Second, using the cleaned data, I then created a dataset called team data which contains 27 unique values. 
Here is a link for the column names and descriptions of this team dataset: https://github.com/djs6478/Capstone-Premier-League-Prediction/blob/main/team_data/%23%20Column%20Names%20and%20Descriptions%20for%20Premier%20League.md
Third, using the cleaned data I created another dataset called table. The table data set has the final table(standings) for every year. This includes the wins, draws, and losses at home and away for every team. In the table dataset, I also include data for the Championship League. 






## Modeling

First I created two simple models. The first model I created a poisson model based on home and away goals.This model uses home and away scoring to predict results.
The second model I created used the attack strength and defence strength to find the expected goals by a poisson distribution.
This model was inspired by a reference, the link to this reference can be found here: https://www.pinnacle.com/en/betting-articles/Soccer/how-to-calculate-poisson-distribution/MD62MLXUMKMXZ6A8. 

Second, I used a backwards elimination class to find the best model. To see where I learned to use backwards elimination, checkout these two links here: 
1:https://www.analyticsvidhya.com/blog/2021/04/backward-feature-elimination-and-its-implementation/
 2:https://github.com/Sushil-Deore/Automated_ML/blob/df123e7905e78ef50a8ec56538e2c17e584c5048/Regression_Models/Regression.py

Third, I then built a PredictGoals class. This predicts a teams goals and a teams rivals goals, based on the poisson models I built. Then using the prediction for goals, a probability is calculated for each game. There is the probability to win, lose or draw. This prediction is done for every single team and every single game. Based on the probabilties, a random outcome is generated for each game. After every game has been simulated, a final table is created to show the seasons simulated results. 

## Evaluation

To calculate the probability of a team's placement, I ran the simulation 200 times and collected the results. I gathered the percentage of times every team placed for each position in the table. The simulation I used was for the 2018-2019 season of the Premier League. I then compared these probabilities to the actual results of the 2018-2019 season. The 3 teams with the highest probability to win(Liverpool, Man City and Chelsea) all finished in the top 3. Out of the bottom 4 teams with the worst probabilities, 3 were in relegated. One big mistake in my model is that simulations over predicted that Burnley would almost always finish in last place. They finished 15th instead. 

## Conclusion

We reccommend that this model is only used for the table prediction gambling before the year starts. It's strength is that it is good at predicting the final table based on the performance of previous years. However it doesn't take into account short term factors that could impact the result of a specific game. So it would not be very useful in deciding the probability of the outcome of specific games. 

## Future Work

I would like to use much more data in my prediction. Currently the model only uses goals and goals conceded depending on if teams are home or away and uses the same data for the opposing team. It doesn't account for recent transfer signings that may improve a team. It doesn't account for injuries. There is a alot of data that affects games that isn't used in this model. 

## Presentation Link 

[Presentation](https://github.com/djs6478/Capstone-Premier-League-Prediction/blob/main/Capstone%20Presentation.pdf)

## Repository Navigation

```
├── README.md                    <- The top-level README for reviewers of this project. 
├── create Data                         <- Run this to create the data folders. 
├── simple model                    <- Folder containing the 2 simple models used for the modeling.
├── Modeling                          <- Where all the final models are created. 
├── .gitignore                   <- Plain text file where each line contains a pattern for files/directories to ignore.
├── final_notebook.ipynb         <- Final Jupyter notebook for this project, containing the simulations. 
└── presentation.pdf             <- PDF of the presentation slides for this project.                 