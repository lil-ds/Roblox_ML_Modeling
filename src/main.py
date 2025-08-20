import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    # Load dataset
    roblox = pd.read_csv('/home/lily/github/Roblox_ML_Modeling/data/processed/roblox_dataset_cleaned.csv')

    # Define features and target
    features = ['active', 'favorites', 'serversize', 'likes', 'dislikes',
       'like_ratio', 'player_to_visit_ratio',
       'genre_action', 'genre_adventure', 'genre_education',
       'genre_entertainment', 'genre_obby_&_platformer',
       'genre_party_&_casual', 'genre_puzzle', 'genre_rpg',
       'genre_roleplay_&_avatar_sim', 'genre_shooter', 'genre_shopping',
       'genre_simulation', 'genre_social', 'genre_sports_&_racing',
       'genre_strategy', 'genre_survival', 'camera_not_supported',
       'camera_supported', 'vc_not_supported', 'vc_supported'
    ]
    target = 'visits'

    X = roblox[features]
    y = roblox[target]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nModel Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"R-squared (R²): {r2:.4f}")

if __name__ == "__main__":
    main()
