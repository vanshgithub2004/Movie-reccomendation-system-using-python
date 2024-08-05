import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Step 1: Load the data into a pandas DataFrame
data = pd.read_csv('ratings.csv')

# Take a look at the first few rows of the data
print(data.head())

# Step 2: Define a Reader object
reader = Reader(rating_scale=(1, 5))

# Step 3: Load the data into Surprise Dataset
surprise_data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

# Step 4: Split the data into training and testing sets
trainset, testset = train_test_split(surprise_data, test_size=0.25)

# Step 5: Use the SVD algorithm
model = SVD()

# Step 6: Train the model on the training data
model.fit(trainset)

# Step 7: Predict ratings for the testset and compute RMSE
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

# Step 8: Function to get top N recommendations
def get_top_n_recommendations(user_id, n=10):
    # Get a list of all movie IDs
    all_movie_ids = data['movieId'].unique()
    
    # Get a list of movie IDs that the user has already rated
    rated_movie_ids = data[data['userId'] == user_id]['movieId'].unique()
    
    # Get a list of movie IDs that the user hasn't rated yet
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
    
    # Predict ratings for all unrated movies
    predictions = [model.predict(user_id, movie_id) for movie_id in unrated_movie_ids]
    
    # Sort the predictions by estimated rating in descending order
    sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
    
    # Get the top N recommendations
    top_n_recommendations = sorted_predictions[:n]
    
    return [(pred.iid, pred.est) for pred in top_n_recommendations]

# Example usage
user_id = 1
recommendations = get_top_n_recommendations(user_id, n=10)
print(f"Top 10 recommendations for user {user_id}: {recommendations}")
