"""
Author: Inigo Hohmeyer
Date: May 9th, 2022
Programming Assignment 4: Naive Bayes Text Classification
"""
import copy
import math
import re
import sys

N = 5

#   this function updates the current
#   dictionary with the counts from the current dictionary
def update(current_dict, overall):
    for i in current_dict:
        if i in overall:
            overall[i] += 1
        else:
            overall[i] = 1

#
stop_words = {}
with open("stopwords.txt") as new_file:
    lines = new_file.readlines()
    for i in lines:
        for j in i.split():
            stop_words[j] = 1


#   Learning Phase
with open("tinyCorpus.txt") as file:
    switch = 0
    count = 0
    #   switch = 0, then Person
    #   switch = 1, then Category
    #   switch = 2, then description
    #   switch = 3, we already read the description we are just waiting until
    #   we reach a blank space
    #   If blank then reset to 1 and if blank do not add to the category.
    #   If blank and count is equal to N then we stop
    corpus = file.readlines()
    cat_count = {}
    current_cat = ""
    final_index = 0
    for index, i in enumerate(corpus):
        #   This means we have reached the end of the corpus
        if count == N and i == "\n":
            break
        #   This means that we are in still in the in-between space
        elif switch == 0 and i == "\n":
            continue
        #   We are at the person's name
        elif switch == 0:
            #   Updates the number of biographies we've gone over
            count += 1
            switch += 1
        #   This means that we are at the category
        elif switch == 1:
            #   If the category has already been created
            if i.split()[0] in cat_count:
                cat_count[i.split()[0]][0] += 1
                switch += 1
                current_cat = i.split()[0]
            else:
                cat_count[i.split()[0]] = [1, {}]
                switch += 1
                current_cat = i.split()[0]
        #   This means we were in the description and
        #   now we have reached the in-between space
        elif switch == 2 and i == "\n":
            switch = 0
        #   this means that we are in the description
        #   but we will only do the description once
        elif switch == 2:
            desc_line = index
            #   creates a dictionary which will be only used for this autobiography
            current_dict = {}
            #   this will iterate through the description
            while corpus[desc_line] != "\n":
                desc_val = re.sub(r'[^\w\s]', '', corpus[desc_line])
                for j in desc_val.split():
                    if j.lower() not in current_dict and j.lower() not in stop_words and len(j) > 2:
                        current_dict[j.lower()] = 1
                desc_line += 1
            update(current_dict, cat_count[current_cat][1])
            #   switches to 3 so we are in the description
            switch = 3
        #   This means that we have reached the end of the description
        elif switch == 3 and i == "\n":
            switch = 0
        final_index = index + 1
freq_table = copy.deepcopy(cat_count)

#   learning phase
for i in freq_table:
    freq_table[i][0] = -math.log2((cat_count[i][0]/N + 0.1)/(1 + len(cat_count) * 0.1))
    for j in freq_table[i][1]:
        freq_table[i][1][j] = -math.log2((cat_count[i][1][j]/cat_count[i][0] + 0.1)/(1 + 2 * 0.1))
print(freq_table)
# 3.2 Applying the classifier to the training data.
pred_dict = {}
with open("tinyCorpus.txt") as file:
    switch = 0
    test_corpus = file.readlines()
    #   the index starts at 0
    #   that seems to pe a problem
    for value in test_corpus[final_index:]:
        index = final_index
        #   we were in the in between zone
        #   we've reached a name
        if switch == 0 and value != "\n":
            pred_dict[value] = ["", {}]
            current_bio = value
            switch = 1
        #   if switch is equal to 1
        #   this means that we are at
        #   category
        #   we will put this as the true category
        elif switch == 1:
            pred_dict[current_bio][0] = value.split()[0]
            switch = 2
        #   this means that we have reached the description
        elif switch == 2:
            #   this puts the probability of each category into the dictionary
            for i in freq_table:
                pred_dict[current_bio][1][i] = freq_table[i][0]
                #   resets the line
                line = final_index
            #   resets the dictionary so there are no repeats
                repeat_dict = {}
            #   goes through the description
                while test_corpus[line] != "\n":
                    #   takes out the punctuation in each line
                    no_punc_line = re.sub(r'[^\w\s]', '', test_corpus[line])
                    #   goes through the line
                    for j in no_punc_line.split():
                        #   goes through the category
                        #   if it's in the dictionary of the category and we have not seen it before
                        #   in this biography then we add its value
                        if j.lower() in freq_table[i][1] and j.lower() not in repeat_dict:
                            print("adding", j.lower(), "for", current_bio, "in", i)
                            print("")
                            pred_dict[current_bio][1][i] += freq_table[i][1][j.lower()]
                            repeat_dict[j.lower()] = 1
                    line += 1
            switch = 3
        elif switch == 3 and value == "\n":
            switch = 0
        final_index += 1
#   Prediction dictionary has the Biography: Predicted Category, [L(C|B) for each category]
def recoverProb(pred):
    cat_prob = copy.deepcopy(pred)
    m = sys.maxsize
    pred_cat = ""
    for i in pred:
        for j in pred[i][1]:
            if pred[i][1][j] < m:
                m = pred[i][1][j]
                pred_cat = j
    for i in pred:
        for j in pred[i][1]:
            if pred[i][1][j] - m < 7:
                cat_prob[i][1][j] = pow(2, (m-pred[i][1][j]))
            else:
                cat_prob[i][1][j] = 0
    for i in cat_prob:
        total = sum(cat_prob[i][1].values())
        for j in cat_prob[i][1]:
            cat_prob[i][1][j] = cat_prob[i][1][j]/total
    return cat_prob

final_prob = recoverProb(pred_dict)
def printOutput(final):
    for i in final:
        min = -sys.maxsize
        prediction = ""
        for j in final[i][1]:
            if final[i][1][j] > min:
                prediction = j
                min = final[i][1][j]
        if prediction == final[i][0]:
            print(i.strip(), "Prediction:", prediction, "Right" )
        else:
            print(i.strip(), "Prediction:", prediction, "Wrong")
        for n in final[i][1]:
            print(n, ":", final[i][1][n], end=" ")
        print("\n")
printOutput(final_prob)
























