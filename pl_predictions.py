import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score

matches=pd.read_csv("matches.csv",index_col=0)

matches["date"]=pd.to_datetime(matches["date"])

matches["venue_code"]=matches["venue"].astype("category").cat.codes

matches["opp_code"]=matches["opponent"].astype("category").cat.codes

matches["hour"]=matches["time"].str.replace(":.+","",regex=True).astype("int")

matches["day_code"] = matches["date"].dt.dayofweek

# Selects each respective result and if it is a win or false it would save 1 or 0.
matches["target"]=(matches["result"]=="W".upper()).astype("int")

# Random Forest machine learning algorithm. It would get non-linear patterns.

# n_estimators: This parameter determines the number of trees in the random forest. In this case, it's set to 50, meaning the random forest will consist of 50 decision trees.

# min_samples_split: This parameter sets the minimum number of samples required to split an internal node. If a node has fewer samples than this value, it will not be split. In this case, it's set to 10.

# random_state: This parameter is used to seed the random number generator. It ensures reproducibility, meaning that if you run the same code with the same random_state, you'll get the same results. Here, it's set to 1.
rf=RandomForestClassifier(n_estimators=50,min_samples_split=10, random_state=1)

# Train with past data.
trained_data=matches[matches["date"]<"2022-01-01"]

test_data=matches[matches["date"]>"2022-01-01"]

predictors=["venue_code","opp_code","hour","day_code"]

rf.fit(trained_data[predictors],trained_data["target"])

preds=rf.predict(test_data[predictors])

acc= accuracy_score(test_data["target"],preds) #To check the accuracy.  

combined=pd.DataFrame(dict(actual=test_data["target"],prediction=preds))

print(pd.crosstab(index=combined["actual"],columns=combined["prediction"]))

precision_score(test_data["target"],preds)

grouped_matches=matches.groupby("team")
group=grouped_matches.get_group("Manchester City")

def rolling_averages(group,cols,new_cols):
    group=group.sort_values("date")
    # Check the form of the club in the last three games based on the stats in (col). The closed="left" ensures that it doesn't consider itslf.
    rolling_stats=group[cols].rolling(3,closed="left").mean()
    group[new_cols]=rolling_stats
    group =group.dropna(subset=new_cols)
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

rolling_averages(group, cols, new_cols)

matches_rolling =matches.groupby("team").apply(lambda x:rolling_averages(x,cols,new_cols))

matches_rolling.droplevel('team')

matches_rolling.index=range(matches_rolling.shape[0])

def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

combined, precision = make_predictions(matches_rolling, predictors + new_cols)


combined=combined.merge(matches_rolling[["date","team","opponent","result"]],left_index=True,right_index=True)


class MissingDict(dict): 
    __missing__ = lambda self, key: key

map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd", "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"} 
mapping = MissingDict(**map_values)

combined["new_team"] = combined["team"].map(mapping)

merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])

final_preds=merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()

print(final_preds)