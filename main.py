from chatbot_core import ChatBot, start_socket
import random
import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import codecs
from utils import filter_text


class MyBot(ChatBot):
    def __init__(self, socket, domain, user, password, config_file='bot_config.json', on_server=True):
        super(MyBot, self).__init__(socket, domain, user, password)
        self.on_server = on_server
        self.last_search = None
        with open(config_file) as config:
            self.bot_config = json.load(config)
        self.vectorizer, self.intent_example_vectorized, self.classifier = self.init_classifier()
        self.qa_dataset = dict()
        # TODO: prevent repeating request
        # self.response_cache = dict()  # {request:[response,time_created]}
        # self.cache_time = 15 * 60  # cache is valid for 15 minutes

    def init_classifier(self):
        """
        """
        intent_example = []
        intents = []
        for intent, intent_data in self.bot_config['intents'].items():
            for example in intent_data['examples']:
                intent_example.append(example)
                intents.append(intent)

        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        """ Other vectorizer examples:
            vectorizer = TfidfVectorizer(analyzer='char_wb')
            vectorizer = CountVectorizer()
        """
        intent_example_vectorized = vectorizer.fit_transform(intent_example)
        classifier = LinearSVC().fit(intent_example_vectorized, intents)

        return vectorizer, intent_example_vectorized, classifier

    def get_intent(self, shout, threshold=0.4):
        """
        Returns best matching intent for shout;
        currently using greedy strategy to match intent
        :param threshold: maximal accepting distance threshold
        :param shout: Input shout
        :return: Best matching intent
        """
        question_vector = self.vectorizer.transform([shout])
        intent = self.classifier.predict(question_vector)[0]

        examples = self.bot_config['intents'][intent]['examples']
        for example in examples:
            dist = nltk.edit_distance(shout, example)
            dist_percentage = dist / len(example)
            if dist_percentage < threshold:
                return intent

    def get_answer_by_intent(self, intent, algorithm='random'):
        """
        Picks phrase for input intent using selected algorithm
        :param algorithm: Chosen algorithm
        :param intent: Input intent
        :return: Chosen phrase
        """
        if intent in self.bot_config['intents']:
            phrases = self.bot_config['intents'][intent]['responses']
            if not algorithm or algorithm == 'random':
                return random.choice(phrases)

    def load_dataset(self, source_file='train.json'):

        with open(source_file) as dataset_file:
            self.qa_dataset = json.load(dataset_file)

            print(self.qa_dataset)

        dialogues = [(dialogue_line['question'], dialogue_line['answer']) for dialogue_line in self.qa_dataset]

        questions = set()
        qa_dataset = []

        for replicas in dialogues:
            if len(replicas) < 2:
                continue

            # remove /n, ? and .
            question = filter_text(replicas[0][2:])
            answer = replicas[1][2:]

            if question and question not in questions:
                questions.add(question)
                qa_dataset.append([question, answer])

        qa_by_word_dataset = {}  # {'word': [[q, a], ...]}
        for question, answer in qa_dataset:
            words = question.split(' ')
            for word in words:
                if word not in qa_by_word_dataset:
                    qa_by_word_dataset[word] = []
                qa_by_word_dataset[word].append((question, answer))

        qa_by_word_dataset_filtered = {word: qa_list
                                       for word, qa_list in qa_by_word_dataset.items()
                                       if len(qa_list) < 1000}
        return qa_by_word_dataset_filtered

    def generate_answer_by_text(self, text, consider_last_n=1000, min_result_threshold=0.2):
        text = filter_text(text)
        words = text.split(' ')
        qa = []
        qa_by_word_dataset_filtered = self.load_dataset()
        for word in words:
            if word in qa_by_word_dataset_filtered:
                qa += qa_by_word_dataset_filtered[word]
        qa = list(set(qa))[:consider_last_n]

        results = []
        for question, answer in qa:
            dist = nltk.edit_distance(question, text)
            dist_percentage = dist / len(question)
            results.append([dist_percentage, question, answer])

        if results:
            dist_percentage, question, answer = min(results, key=lambda pairs: pairs[0])
            if dist_percentage < min_result_threshold:
                return answer

    def get_failure_phrase(self, algorithm='random'):
        return random.choice(self.bot_config['failure_phrases']) if algorithm and algorithm == 'random' else None

    def ask_chatbot(self, user, shout, timestamp):
        """
        Handles an incoming shout into the current conversation
        :param user: user associated with shout
        :param shout: text shouted by user
        :param timestamp: formatted timestamp of shout
        """
        resp = f""  # Generate some response here
        intent = self.get_intent(shout)

        # We are seaching ready answer
        if intent:
            resp = self.get_answer_by_intent(intent)

        # Generate eligible answer by text
        if not resp:
            resp = self.generate_answer_by_text(shout)

        # Failed to find a response
        if not resp:
            resp = self.get_failure_phrase()

        if self.on_server:
            self.propose_response(resp)
        else:
            return resp

    def ask_appraiser(self, options):
        """
        Selects one of the responses to a prompt and casts a vote in the conversation
        :param options: proposed responses (botname: response)
        """
        selection = random.choice(list(options.keys()))
        self.vote_response(selection)

    def ask_discusser(self, options):
        """
        Provides one discussion response based on the given options
        :param options: proposed responses (botname: response)
        """
        selection = list(options.keys())[0]  # Note that this example doesn't match the voted choice
        self.discuss_response(f"I like {selection}.")

    def on_discussion(self, user: str, shout: str):
        """
        Handle discussion from other subminds. This may inform voting for the current prompt
        :param user: user associated with shout
        :param shout: shout to be considered
        """
        pass

    def on_login(self):
        """
        Do any initialization after logging in
        """
        pass


if __name__ == "__main__":
    # Testing
    bot = MyBot(start_socket("2222.us"),
                "Private",
                "testrunner",
                "testpassword",
                on_server=False)
    while True:
        try:
            utterance = input('[In]: ')
            response = bot.ask_chatbot(f'', utterance, f'')
            print(f'[Out]: {response}')
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    # Running on the forum
    MyBot(start_socket("2222.us", 8888), f"chatbotsforum.org", None, None, True)
    while True:
        pass
