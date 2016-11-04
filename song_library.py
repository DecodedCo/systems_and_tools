
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
    return lyrics_data["GetLyricResult"]["Lyric"]#.replace("\n"," ").replace(","," ")


def getLyricsWithWord(word):
    # create url that songs mentioning the word
    songs_list_url = 'http://api.chartlyrics.com/apiv1.asmx/SearchLyricText?lyricText=' + word
    # read the xml from the url
    songs_parsed_xml = readXML(songs_list_url)
    # extract the song details from the parsed xml
    songs_details = getSongDetails(songs_parsed_xml)
    # the endpoint that will have lyrics
    lyrics_rootURL = "http://api.chartlyrics.com/apiv1.asmx/GetLyric?"
    # create variable/object where we will store the output
    lyricsObject = dict()
    # initialise keys
    lyricsObject["lyrics"] = list()
    lyricsObject["song"] = list()
    lyricsObject["artist"] = list()
    # Iterate over each song, in order to obtain lyrics
    for song in songs_details:
        # create url that has the lyrics
        url = lyrics_rootURL + "LyricChecksum=" + song[0] + "&LyricId=" + song[1]  
        # read the lyrics from the url
        lyrics = getLyrics(url)
        # save the new data
        lyricsObject["lyrics"].append(lyrics)
        lyricsObject["song"].append(song[2])
        lyricsObject["artist"].append(song[3])

    return lyricsObject

def update(big_object, small_object):
    if "song" in big_object:
        big_object["song"] += small_object["song"]
        big_object["artist"] += small_object["artist"]
        big_object["lyrics"] += small_object["lyrics"]
    else:
        big_object = small_object
    return big_object

def makePlot(pairwise_similarity, all_songs, colours):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    #discrete color scheme
    cMap = ListedColormap(colours)

    # #data
    # np.random.seed(42)
    # data = np.random.rand(4, 4)
    data = pairwise_similarity.toarray()
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=cMap)

    #legend
    cbar = plt.colorbar(heatmap)

    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['$0$','$1$','$2$','$>3$']):
        cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center')
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('# of contacts', rotation=270)


    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    #lebels
    # column_labels = list('ABCD')
    column_labels = all_songs["song"]
    row_labels = column_labels
    # row_labels = list('WXYZ')
    ax.set_xticklabels(column_labels, minor=False, rotation=90)
    ax.set_yticklabels(row_labels, minor=False)

    plt.show()

def convertToString(lyrics):
    return " ".join(lyrics)

# remove punctuation to make autofill more sensible
def clean_text(text):
    # remove new lines and commans
    text = text.replace("\n"," ").replace(","," ")
    words = []
    for word in text.split(" "):
        # remove apostrphes and anything that follows
        new_word = word.split("'")[0]
        words.append(new_word)
    return " ".join(words)

# Uses autocomplete library to select a next word starting with a 
# particular letter, considering the training set
def nextWord(word, letter, all_lyrics):
    # read in libraries
    import autocomplete
    from autocomplete import models
    # covert all lyrics to one long string
    lyrics_string = convertToString(all_lyrics)
    # train the model on the lyrics
    models.train_models(lyrics_string)
    # make the prediction
    return autocomplete.predict(word,letter)


# Use autocomplete to predict n words after the starting word, using each new word
# as the starting point for the enxt
def create_lyrics(starting_word, length, lyrics):
    training_string = " ".join(lyrics)
    # remove existing autocomplete model
    os.system("rm models_compressed.pkl")
    # clean the text
    new_training_string = clean_text(training_string)
    models.train_models(new_training_string) # requires sudo
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    output = list()
    curr_length = 1
    output.append(starting_word)
    curr_word = starting_word
    # iterate until enough next words have been selected
    while(curr_length < length):
        letters_count = dict()
        # iterate over each candidate next word in the alphabet
        for letter in alphabet:
            # look at possible next words starting from the current letter
            possible_next_words = autocomplete.predict(curr_word,letter)
            # if there's many possibilities, select one in a weighted fashion
            if len(possible_next_words) > 0:
                n = len(possible_next_words)
                counts = [x[1] for x in possible_next_words]
                #print counts
                count_sum = np.sum(counts)
                #print count_sum
                weights = [x[1]*1.0/count_sum for x in possible_next_words]
                index = choice(range(0,n), 1, p=weights)[0]
                index = randint(0,n-1) # equal weights means less likely to get stuck in loop
                next_word = possible_next_words[index][0]
                next_word_count = possible_next_words[0][1]
                if next_word not in letters_count:
                    letters_count[next_word] = next_word_count
                else:
                    if next_word_count > letters_count[next_word]:
                        letters_count[next_word] = next_word_count
        # predict the word that has the highest count                    
        predicted_word = max(letters_count.iteritems(), key=operator.itemgetter(1))[0]
        # if this new word has appeared in our lyrics too much already, resort to hard reset
        if output.count(predicted_word) > 5:
            predicted_word = choice(["and","the","or","if","to","you","always","forever"], 1, p=[1.0/8,1.0/8,1.0/8,1.0/8,1.0/8,1.0/8,1.0/8,1.0/8])[0]
        output.append(predicted_word)
        curr_length += 1
        curr_word = predicted_word
    # Write lyrics to file
    open( "mylyrics.txt", "wb" ).write(" ".join(output))
    return " ".join(output )