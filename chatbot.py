# Portfolio 7: Chatbot
# Names: Shreya Valaboju, Soham Mukherjee
import json

# Description of project

# import libraries
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

nltk.download('punkt')
from nltk.corpus import stopwords
import re
from pprint import pprint

players_info = {}


def printKnowledgeBase(fp_name):
    players_pickle = open(fp_name, "rb")
    players_info = pickle.load(players_pickle)
    # pprint(players_info)
    return players_info


# use cosine similarity and tf-idf vectorization to match a player to what the user input is or who the user is asking about
def train(words):
    # DEFAULT topic is "Born"
    topic_keywords = ["Born", "College", "NBA draft", "High school", "Listed weight", "Listed height", "Position",
                      "Playing career"]
    topic_string = ' '.join(topic_keywords)
    similarity_scores_names = []
    player_names = list(players_info.keys())
    user_query = ' '.join(words)

    # Q: Where was Luka Born? --> Where Luka Born
    # find the closest name to what the user is asking about
    for name in player_names:
        name = name.replace("_", " ")
        vectorizer = TfidfVectorizer().fit_transform([user_query, name])
        similarity_scores_names.append(cosine_similarity(vectorizer[0], vectorizer[1])[0][0])

    most_similar_player_idx = similarity_scores_names.index(max(similarity_scores_names))
    player_name = player_names[most_similar_player_idx]

    # find similarity between user query and topic_keywords
    similarity_scores = []
    for topic in topic_keywords:
        vectorizer = TfidfVectorizer().fit_transform([user_query, topic])
        similarity_scores.append(cosine_similarity(vectorizer[0], vectorizer[1])[0][0])

    most_similar_topic_idx = similarity_scores.index(max(similarity_scores))
    topic = topic_keywords[most_similar_topic_idx]

    # find the player and the related topic.
    if topic in players_info[player_name]:
        print("Here's some information about", player_name)
        pprint(players_info[player_name][topic])
    else:
        print("Could not find information about " + player_name + " and their " + topic + "; it may not exist. Try "
                                                                                          "again?")


# lemmatize, remove stop words, numbers, lowercase
def preprocess(text_in):
    lemma = nltk.stem.WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text_in = re.sub(r'[^\w\s]', '', text_in)  # remove punctuation
    word_tokens = nltk.word_tokenize(text_in.lower())  # tokenize into words
    word_tokens = [word for word in word_tokens if word not in stop_words]
    word_tokens = [lemma.lemmatize(word) for word in word_tokens]

    # get player info by using cosine-similarity to find most similar player in user-input
    train(word_tokens)


def load_users():
    # load existing user data from file, or initialize new file if it doesn't exist
    try:
        with open("users.json", "r") as f:
            users = json.load(f)
    except FileNotFoundError:
        users = {}
    return users


def update_user(users):
    with open("users.json", "w") as f:
        json.dump(users, f)


def chat():
    users = load_users()

    print("Hello! My name is Champ, you can ask me any anything about the current players on the Dallas Mavs!")
    print(
        "You can type in a player's name to get information OR ask something specific about a player currenty on the team.")
    print("Type 'quit' to end to stop chatting.")
    user_name = input("Enter your name: ")

    if user_name in users:
        user_info = users[user_name]
    else:
        user_info = {"name": user_name, "personal_info": {}, "likes": [], "dislikes": []}
        users[user_name] = user_info

    print("Champ: Howdy", user_name, "! Go ahead, ask me question")

    # Get personalized remarks from the user model
    if user_info["likes"]:
        print(f"Champ: By the way {user_name}, I remember you said you like {', '.join(user_info['likes'])}.")
    if user_info["dislikes"]:
        print(f"Champ: Also, I remember you said you don't like {', '.join(user_info['dislikes'])}.")

    while True:
        user_input = input(user_name + ": ")
        if 'quit' in user_input.lower():
            break
        else:
            print("Champ: Ok, let me get you that information...")
            preprocess(user_input)  # preprocess user questions

            # Update user model based on user's response
            print(f"{user_name}, is there anything else you'd like me to know about you?")
            new_info = input("Likes or dislikes, specifically? ")
            if 'like' in new_info.lower().split():
                user_info["likes"].append(new_info.split('like ')[-1])
            elif 'dislike' in new_info.lower().split():
                user_info["dislikes"].append(new_info.split('dislike ')[-1])
    update_user(users)
    print("Thanks for chatting,", user_name, "! I hope I answered all your questions, and always - Go Mavs! :)")


if __name__ == '__main__':
    players_info = printKnowledgeBase("players.p")
    chat()
