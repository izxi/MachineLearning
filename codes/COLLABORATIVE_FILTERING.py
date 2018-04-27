# A simple insight into Collaborative Filtering

# a simple database containing users, movies they have watched, and the ratings.
data = {"Jacob": {"The Avengers": 3.0, "The Martin": 4.0, "Guardians of the Galaxy": 3.5, "Edge of Tomorrow": 5.0, "The Maze Runner": 3.0},
     "Jane": {"The Avengers": 3.0, "The Martin": 4.0, "Guardians of the Galaxy": 4.0, "Edge of Tomorrow": 3.0, "The Maze Runner": 3.0, "Unbroken": 4.5},
     "Jim": {"The Martin": 1.0, "Guardians of the Galaxy": 2.0, "Edge of Tomorrow": 5.0, "The Maze Runner": 4.5, "Unbroken": 2.0}}
# obtain the movies watched and ratings by using the get function and specifying the user.
data.get("Jacob")
# find the movies that have been watched by Jacob and Jane by using the intersection of both sets.
#   ['The Maze Runner', 'Edge of Tomorrow', 'The Avengers', 'The Martin', 'Guardians of the Galaxy']
common_movies = list(set(data.get("Jacob")).intersection(data.get("Jane")))
#   ['The Maze Runner', 'Edge of Tomorrow', 'Guardians of the Galaxy', 'The Martin']
common_movies2 = list(set(data.get("Jacob")).intersection(data.get("Jim")))
# find possible recommendations by finding the movies that Jane has watched and the ones Jacob has not
# Some recommendation systems will use Collaborative Filtering to looking for people that rate movies similar to pull recommendations from their watched list.
recommendation = list(set(data.get("Jane")).difference(data.get("Jacob"))) # ['Unbroken']
recommendation2 = list(set(data.get("Jacob")).difference(data.get("Jim")))  # ['The Avengers']

# similar_mean function to compute the average difference in rating
# This will tell if ratings on the movies are similar, or not. 
# We'll have threshold of 1 rating or less to consider the recommendation an adequate one.
def similar_mean(same_movies, user1, user2, dataset):
    total = 0
    for movie in same_movies:
        total += abs(dataset.get(user1).get(movie) - dataset.get(user2).get(movie))
    return total/len(same_movies)

print(similar_mean(common_movies, "Jacob", "Jane", data)) # 0.5   The recommendation of 'Unbroken' should be a good watch for Jacob!
print(similar_mean(common_movies2, "Jacob", "Jim", data)) # 1.5   Since the average difference in rating is larger than our 1.0 threshold, 'The Avengers' would not be a good recommendation for Jim.

