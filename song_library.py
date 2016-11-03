
import operator
from random import randint
import autocomplete
from autocomplete import models
import os
from numpy.random import choice
import urllib2
import xmltodict
import numpy as np

def readXML(url):
	file = urllib2.urlopen(url)
	data = file.read()
	file.close()
	data = xmltodict.parse(data)
	return data

def getSongDetails(parsed_xml):
	songs_details = [[song["LyricChecksum"],song["LyricId"],song["Song"],song["Artist"]] 
		for song in parsed_xml["ArrayOfSearchLyricResult"]["SearchLyricResult"] if "LyricId" in song]
	return songs_details

def getLyrics(url):
    file = urllib2.urlopen(url)
    lyrics_data = file.read()
    file.close()
    lyrics_data = xmltodict.parse(lyrics_data)
    return lyrics_data["GetLyricResult"]["Lyric"].replace("\n","").replace(","," ")

def clean_text(text):
    words = []
    for word in text.split(" "):
        new_word = word.split("'")[0]
        words.append(new_word)
    return " ".join(words)

def create_lyrics(starting_word, length, training_string):
    os.system("rm /Library/Python/2.7/site-packages/autocomplete/models_compressed.pkl")
    new_training = clean_text(training_string)
    models.train_models(new_training)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    output = list()
    curr_length = 1
    output.append(starting_word)
    curr_word = starting_word
    while(curr_length < length):
        letters_count = dict()
        for letter in alphabet:
            possible_next_words = autocomplete.predict(curr_word,letter)
            if len(possible_next_words) > 0:
                n = len(possible_next_words)
                counts = [x[1] for x in possible_next_words]
                #print counts
                count_sum = np.sum(counts)
                #print count_sum
                weights = [x[1]*1.0/count_sum for x in possible_next_words]
                index = choice(range(0,n), 1, p=weights)[0]
                index = randint(0,n-1)
                next_word = possible_next_words[index][0]
                next_word_count = possible_next_words[0][1]
                if next_word not in letters_count:
                    letters_count[next_word] = next_word_count
                else:
                    if next_word_count > letters_count[next_word]:
                        letters_count[next_word] = next_word_count
                        
        predicted_word = max(letters_count.iteritems(), key=operator.itemgetter(1))[0]
        if output.count(predicted_word) > 5:
            predicted_word = choice(["and","the","or","if","to","you","always","forever"], 1, p=[1.0/8,1.0/8,1.0/8,1.0/8,1.0/8,1.0/8,1.0/8,1.0/8])[0]
        output.append(predicted_word)
        curr_length += 1
        curr_word = predicted_word
    #outfile.write("\n".join(itemlist))
    open( "mylyrics.txt", "wb" ).write(" ".join(output))
    return output 