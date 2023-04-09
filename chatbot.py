# Portfolio 7: Chatbot
# Names: Shreya Valaboju, Soham Mukherjee

# Description of project

#import libraries
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
nltk.download('punkt')
from nltk.corpus import stopwords
import re
from pprint import pprint


players_info={}



def printKnowledgeBase(fp_name):
    players_pickle = open(fp_name, "rb")
    players_info = pickle.load(players_pickle)
    #pprint(players_info)
    return players_info

# use cosine similarity and tf-idf vectorization to match a player to what the user input is or who the user is asking about
def train(words):

    # DEFAULT topic is "Born"
    topic_keywords =["Born","College","NBA draft","High school","Listed weight","Listed height","Position","Playing career"]
    topic_string = ' '.join(topic_keywords)
    similarity_scores_names = []
    player_names = list(players_info.keys())
    user_query = ' '.join(words)

    # Q: Where was Luka Born? --> Where Luka Born
    # find the closest name to what the user is asking about
    for name in player_names:
        name = name.replace("_"," ")
        vectorizer = TfidfVectorizer().fit_transform([user_query,name])
        similarity_scores_names.append(cosine_similarity(vectorizer[0], vectorizer[1])[0][0])

    most_similar_player_idx = similarity_scores_names.index(max(similarity_scores_names))
    player_name =player_names[most_similar_player_idx]

    # find similarity between user query and topic_keywords
    similarity_scores=[]
    for topic in topic_keywords:
        vectorizer = TfidfVectorizer().fit_transform([user_query, topic])
        similarity_scores.append(cosine_similarity(vectorizer[0], vectorizer[1])[0][0])

    most_similar_topic_idx = similarity_scores.index(max(similarity_scores))
    topic = topic_keywords[most_similar_topic_idx]

    # find the player and the related topic.
    print("Here's some information about", player_name)
    pprint(players_info[player_name][topic])





# lemmatize, remove stop words, numbers, lowercase
def preprocess(text_in):
    lemma = nltk.stem.WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text_in = re.sub(r'[^\w\s]', '', text_in) # remove punctuation
    word_tokens = nltk.word_tokenize(text_in.lower())  # tokenize into words
    word_tokens = [word for word in word_tokens if word not in stop_words]
    word_tokens = [lemma.lemmatize(word) for word in word_tokens]

    # get player info by using cosine-similarity to find most similar player in user-input
    train(word_tokens)


def chat():
    print("Hello! My name is Champ, you can ask me any anything about the current players on the Dallas Mavs!")
    print("You can type in a player's name to get information OR ask something specific about a player currenty on the team.")
    print("Type 'quit' to end to stop chatting." )
    user_name = input("Enter your name: ")

    print("Champ: Howdy",user_name, "! Go ahead, ask me question")

    while True:
        user_input = input(user_name+": ")
        if 'quit' in user_input.lower():
            break
        else:
            print("Champ: Ok, let me get you that information...")
            preprocess(user_input)  # preprocess user questions
            # case 1 (general info abt 1 player) : use cosine similarity or tf-idf to find what exact player they are talking about

            # case 2 (for specific information about a player): find keywords
                # if keywords are like "birth place" or "Luka" -> get Luka's birthplace information



    print("Thanks for chatting,", user_name,"! I hope I answered all your questions, and always - Go Mavs! :)")

if __name__ == '__main__':
    players_info = printKnowledgeBase("players.p")
    chat()


