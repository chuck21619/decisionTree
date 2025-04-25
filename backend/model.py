from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
import xgboost as xgb

def train_model(df, player_columns, le_players):
    X = df[player_columns]
    y = df['winner']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [25, 50, 100],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    }

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le_players.classes_),
        eval_metric='mlogloss'
    )
    
    stratified_kfold = StratifiedKFold(n_splits=3)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=stratified_kfold, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    
    print("Best params:", grid_search.best_params_)
    print("Best accuracy:", accuracy)

    # Log accuracy for all tested models (i.e., the grid search results)
    print("Grid Search Results:")
    for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        print(f"Params: {params} | Mean Test Accuracy: {mean_score}")

    return best_model, accuracy, X
