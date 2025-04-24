from sklearn.model_selection import train_test_split
import xgboost as xgb

def train_model(df, player_columns, le_players):
    X = df[player_columns]
    y = df['winner']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le_players.classes_),
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy, X
