# Portfolio 7: Chatbot
# Names: Shreya Valaboju, Soham Mukherjee


# To -Do's (ADD MORE HERE):
#   - improvements on user model (add age, other info), also check for user name case, like "Shreya" should be 'shreya'
#   - I was thinking about adding emotion/sentiment analysis (we'll see if we have time)
#   -


# import libraries
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import random
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download('punkt', quiet=True)
import re
from pprint import pprint


players_info = {}   # knowledge base with player info
intents_dict={} # basic intents not about players specifically


# print and store the players knowledge base
def printKnowledgeBase(fp_name):
    players_pickle = open(fp_name, "rb")
    players_info = pickle.load(players_pickle)
    # pprint(players_info)
    return players_info



# gets all the synonyms for a specfic topic
def getTopicSynonyms(topics_list):
    synonyms=[]
    syn_dict={}

    # iterate through each topic from the knowledge base
    for topic in topics_list:
        for syn in wordnet.synsets(topic):  # get all synonyms for each topic and put them into a list
            for l in syn.lemmas():
                synonyms.append(l.name())
        syn_dict[topic] = synonyms

    return syn_dict

# use cosine similarity and tf-idf vectorization to match a player to what the user input is or who the user is asking about
def train(words):

    player_names = list(players_info.keys()) # list of player names

    topic_keywords = ["Born", "College", "NBA draft", "High school", "Listed weight", "Listed height", "Position",
                      "Playing career", "Men's basketball", "Nationality", "League"] # topics that we have knowledge on
    topic_dict = getTopicSynonyms(topic_keywords) # topics can be mentioned in a different way, like "born" is also "birth"
    similarity_scores_names = []
    user_query = ' '.join(words)

    # check if the user is just greeting
    if user_query in intents_dict['greet']['patterns']:
        print("Champ: "+ random.choice(intents_dict['greet']['responses']))
        return

    # check if the user is thanking the bot
    elif user_query in intents_dict['thanks']['patterns']:
        print("Champ: " + random.choice(intents_dict['thanks']['responses']))
        return

    # check if the user is saying bye to the bot
    elif user_query in intents_dict['goodbye']['patterns']:
        print("Champ: " + random.choice(intents_dict['goodbye']['responses']))
        return

    # check if the user is asking about the pheonix suns (our rival)
    elif user_query in intents_dict['funny']['patterns']:
        print("Champ: " + random.choice(intents_dict['funny']['responses']))
        return

    # check if the user is asking who is the best player, who is the GOAT
    if user_query in intents_dict['goat']['patterns']:
        print("Champ: " + random.choice(intents_dict['goat']['responses']))
        return

    # check if the user is asking about mark cuban
    if user_query in intents_dict['mark_cuban']['patterns']:
        print("Champ: " + random.choice(intents_dict['mark_cuban']['responses']))
        return

    # check if the user is talking about any of the players we have info on
    new_player_names = [] # holds names in a more readbale way, like "Luka_Doncic" -> "Luka Doncic"
    for n in player_names:
        n = n.split("_")
        if n[0].lower() == 'tim jr.':   #special case for player, Tim Hardaway jr.
            n[0] = 'tim'
        new_player_names.append(n[0].lower())
        new_player_names.append(n[1].lower())

    intersection_players = len([i for i in new_player_names if i in words]) # check if the user mentions any of the players we have info on

    if intersection_players == 0:  # no players in user query, can't understand default
        print("Champ: I'm happy to help, try asking me something about a player.")
        return


    # find the closest player name to what the user is asking about, using cosine similarity and tf-idf vectorizer
    for name in player_names:
        name = name.replace("_", " ")
        vectorizer = TfidfVectorizer().fit_transform([user_query, name])
        similarity_scores_names.append(cosine_similarity(vectorizer[0], vectorizer[1])[0][0])

    most_similar_player_idx = similarity_scores_names.index(max(similarity_scores_names)) # index of most similar player name in list
    player_name = player_names[most_similar_player_idx] # get player name from dictionary

    # find similarity between user query and topic_keywords, try to understand what topic they're talking about
    similarity_scores = []
    topic=""
    topic_synonym_exists = False
    for topic in topic_keywords: # using cosine similarity and tf-idf vectorizer
        vectorizer = TfidfVectorizer().fit_transform([user_query, topic])
        similarity_scores.append(cosine_similarity(vectorizer[0], vectorizer[1])[0][0])

    most_similar_topic_idx = similarity_scores.index(max(similarity_scores)) # index of most similar topic name in list

    # if they're talking about a player, but can't match to a topic
    if all(sim_score == 0 for sim_score in similarity_scores):
        # try to see if a synonym is being said and match to topic
        for t in topic_dict.keys():
            if topic_synonym_exists: # don't go through all topics
                break
            for syn in topic_dict[t]: # for all synonyms
                if syn!=t and syn in words: # found a synonym the user mentions
                    topic_synonym_exists = True
                    topic = t
                    break

        if topic_synonym_exists == False:
            print("Champ: Could not find that specific information about " + player_name)
            return

    if topic_synonym_exists == False: # no synonym, topic said verbatim by the user
        topic = topic_keywords[most_similar_topic_idx]

    # find the player and the related topic and print info found in players dictionary
    if topic in players_info[player_name]:
        print("Champ: Here's some information about", player_name.replace("_"," "), "relating to", topic)
        pprint(players_info[player_name][topic])
    else:
        print("Champ: Could not find information about " + player_name.replace("_"," ") + " and their " + topic + "; it may not exist. Try again?")


# lemmatize, remove stop words, numbers, lowercase
def preprocess(text_in):
    lemma = nltk.stem.WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    word_tokens = nltk.word_tokenize(text_in.lower())  # tokenize into words and lowercase
    word_tokens_l=[]

    for word in word_tokens:    #remove punctuation and try lemmatize player names
        word = re.sub(r'[^\w\s]', '', word)  # remove punctuation
        if word!='': # make sure word is not empty
            word_tokens_l.append(word)

    word_tokens_l = [word for word in word_tokens_l if word not in stop_words] # remove stop words
    word_tokens_l = [lemma.lemmatize(word) for word in word_tokens_l] # lemmatize

    # get player info by using cosine-similarity to find most similar player in user-input
    train(word_tokens_l)


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

# the intents file handles basic things like greetings, goodbye, random questions about mark cuban, etc.
def load_intents(intents_file):
    with open(intents_file, 'r') as f:
        intents_file = json.load(f)

    for i in intents_file['intents']:
        intents_dict[i['tag']] = {
            'patterns': i['patterns'],
            'responses': i['responses']
        }

def chat():

    users = load_users()

    print("Hello! My name is Champ, you can ask me any anything about the current players on the Dallas Mavs!")
    print(
        "You can type in a player's name to get information OR ask something specific about a player currenty on the team.")
    print("I have information about a player's college, high school, height, weight, playing career, draft, and more!")
    print("Type 'quit' to end session and to stop chatting.")
    user_name = input("\nEnter your name: ")

    if user_name in users:
        print(f"Welcome back, {user_name}!")
        user_info = users[user_name]
    else:
        user_info = {"name": user_name, "personal_info": {}, "likes": [], "dislikes": []}
        users[user_name] = user_info

    # Get personalized remarks from the user model
    if user_info["likes"]:
        print(f"Champ: By the way {user_name}, I remember you said you like {', '.join(user_info['likes'])}.")
    if user_info["dislikes"]:
        print(f"Champ: Also, I remember you said you don't like {', '.join(user_info['dislikes'])}.")

    # Update user model based on user's response
    print(f"Champ: {user_name}, is there anything else you'd like me to know about you?")
    new_info = input("Likes or dislikes, specifically? ")

    likes = re.findall(r'(?i)\blike\b\s+((?:(?!\bdislike\b).)+)', new_info)
    dislikes = re.findall(r'(?i)\bdislike\b\s+((?:(?!\blike\b).)+)', new_info)

    for like in likes:
        if like not in user_info["likes"]:
            user_info["likes"].append(like.strip())

    for dislike in dislikes:
        if dislike not in user_info["dislikes"]:
            user_info["dislikes"].append(dislike.strip())

    print("Champ: Howdy", user_name, "! Go ahead, ask me question")
    while True:
        user_input = input(user_name + ": ")
        if 'quit' in user_input.lower():
            break
        else:
            preprocess(user_input)  # preprocess user questions

    update_user(users)
    print("Champ: Thanks for chatting, "+ user_name + "! I hope I answered all your questions, and always - Go Mavs! :)")


if __name__ == '__main__':
    players_info = printKnowledgeBase("players.p")
    load_intents('intents.json')
    chat()
