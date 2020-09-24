
from selenium import webdriver
from bs4 import BeautifulSoup
from collections import Counter
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import time

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# Scrape the desirect transcript
# User must designate desired_speaker and transcript url
# Script also requires Chrome browser and corresponding chromedriver.exe

desired_speaker = "Mike Pence"
url = 'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-press-conference-transcript-april-17'
driver = webdriver.Chrome(executable_path=r'chromedriver.exe')
driver.get(url)
soup = BeautifulSoup(driver.page_source,"lxml")
time.sleep(3)
driver.close()

#Carve out only parts of the transcript spoken by the desired_speaker and combine them in a master_string
#The particular tags used are based on the structure of the transcript webpage

transcript = soup.find('div', {'class':'fl-callout-text'})
master_string = ''
speaker_list = transcript.find_all('p')

for speaker in speaker_list:
    if re.search('Mike Pence:', speaker.text):
        speaker_words = re.sub(r"Mike Pence: \s*\(.*\)\s*", "", speaker.text)
        speaker_words = re.sub(r"\s*\[.*\]\s*", "", speaker_words)
        master_string += speaker_words

#Clean the master_string.
#Expand contracted words
#Convert to lower case

master_string = re.sub(u"(\u2018|\u2019)", "'", master_string)
master_string = expand_contractions(master_string)
master_string = master_string.lower()

#Remove digits from master_string

remove_digits = True
pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
master_string = re.sub(pattern, '', master_string)

#Remove stop words from master_string

stopword_list = stopwords.words('english')
tokens = nltk.word_tokenize(master_string)
tokens = [token.strip() for token in tokens]
tokenized_output = ' '.join([token for token in tokens if token not in stopword_list])


# Tokenize: Split the sentence into words
word_list = nltk.word_tokenize(tokenized_output)


#pos_tagged_word_list = nltk.pos_tag(word_list)

word_list = nltk.pos_tag(word_list)

# Lemmatize list of words and join

lemmatizer = WordNetLemmatizer()

lemmatized_word_list = [lemmatizer.lemmatize(w[0],get_wordnet_pos(w[1])) for w in word_list]

exception_list = ['get','go','u','one']

lemmatized_word_list = [word for word in lemmatized_word_list if word not in exception_list]

# Create a dict to counter words where items are keys and values are word count using collections.Counter

freq = Counter(lemmatized_word_list)

# Generate Word Cloud

x, y = np.ogrid[:300, :300]

mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)

wordcloud = WordCloud(max_words=30,relative_scaling=1,mask=mask,normalize_plurals=False,background_color="white")
wordcloud.generate_from_frequencies(freq)
plt.imshow(wordcloud, interpolation='bilinear')
plt.suptitle("Pence Words")
plt.title("COVID Update April 17th")
plt.axis("off")
plt.show()



