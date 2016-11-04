


import urllib2
import xmltodict


file = urllib2.urlopen('http://api.chartlyrics.com/apiv1.asmx/SearchLyricText?lyricText=something')
data = file.read()
file.close()

data = xmltodict.parse(data)


import song_library as decoded

songs = decoded.getLyricsWithWord("something")

# print songs["song"][0], songs["artist"][0], songs["lyrics"][0]

# Collect songs for 3 different key words

all_songs = dict()
for word in ["something","forever","moment"]:
	songs = decoded.getLyricsWithWord(word)
	all_songs = decoded.update(all_songs, songs)

# Search for NLP compare document similarity python and copy stackoverflow code

from sklearn.feature_extraction.text import TfidfVectorizer

# documents = [open(f) for f in text_files]
tfidf = TfidfVectorizer().fit_transform(all_songs["lyrics"])
# no need to normalize, since Vectorizer will return normalized tf-idf
pairwise_similarity = tfidf * tfidf.T


# print pairwise_similarity[0]

decoded.makePlot(pairwise_similarity, all_songs, ["blue","green","yellow","orange","red"]) # colour from least similar to most similar


# import autocomplete
# from autocomplete import models


# lyrics_string = ' '.join(all_songs["lyrics"])
# # lyrics_string = decoded.convertToString(lyrics)

# models.train_models(lyrics_string)

# print autocomplete.predict('forever','m')

print decoded.nextWord("forever", "m", all_songs["lyrics"])

print decoded.create_lyrics("love",50, all_songs["lyrics"])