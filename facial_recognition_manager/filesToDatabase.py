# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:02:07 2017

@author: Lucas
"""

""" The goal of this program is to convert the data present under the directory 'images_path'
    with the following architecture:
        person-1
        ├── image-1.jpg
        ├── image-2.png
        ...
        └── image-p.png

        ...

        person-m
        ├── image-1.png
        ├── image-2.jpg
        ...
        └── image-q.png
    and the data present under the directory 'London_info' with the following architecture:
        person-1
        ├── info.txt

        ...

        person-m
        ├── info.txt
    The program will save the result under a file 'infos_path' containing the necessary data,
    i.e. an array
        [(encodings, name, path_to_image, profile)]

    To do that, we define here a basic recommender system. We also call for the
    functions of the facialRecognition module.
"""


###############################################################################
# Imports
###############################################################################

# Utilitaries.
import shutil
import os
import re
from os.path import join, split

# Beautiful loading bars.
from tqdm import tqdm

# Image analysis and scientific computations.
import facialRecognition
import numpy as np

# Natural language analysis.
import nltk

###############################################################################
# Main content of the module.
###############################################################################

class basicRecommender:
    """
    A class for basic computation for recommendations.
    We want it to implement 1 function :
        computeProfile(name),
    which assigns a category to the given name.
    The different categories we aim at are by default:
        - Business leader.
        - Techical leader.
        - Digital leader.
        - Digital designer.
        - Digital analyst.
    If the algorithm does not fing a category, it returns the string :
        'Unable to match description with profile'.
    """
    def __init__(self, folder_name_info, fdist = 'default', categories = 'default'):
        """
        Initialization of the class.

        :param folder_name_info: The folder in which all descriptions are stored.
        :param fdist: The frequency distribution of words in the whole description.
        :param categories: The different categories in which to classify people, along with corresponding keywords.
        """
        # Initialize set of info.
        self.folder_name_info = folder_name_info
        # Initialize categories.
        if categories == 'default':
                businessleader_words = ['senior', 'business', 'leader', 'director', 'associate', 'collaboration', 'consultant', 'administrator', 'planner', 'business', 'manager', 'management', 'project', 'projects', 'service']
                technicalleader_words = ['senior', 'scientist', 'director', 'associate', 'policy', 'project', 'projects', 'buildings', 'engineer', 'technical', 'leader', 'transport', 'structure', 'infrastructure']
                digitalleader_words = ['senior', 'scientist', 'digital', 'leader', 'director', 'associate', 'value', 'account', 'enterprise', 'architecture', 'smart', 'cities', 'machine', 'learning']
                digitaldesigner_words = ['com', 'digital', 'designer', 'interaction', 'service', 'design', 'visualisation', 'modelling', 'rhinocerous', 'blender', 'archicad', 'autocad', 'adt', 'revit', 'virtual', 'augmented', 'reality', 'Programming', 'c', 'c++', 'python', 'gui', 'linux', 'open', 'source', 'radiance', 'bimodelling', 'complex', 'geometry', 'facades', '3d', 'printing']
                digitalanalyst_words = ['digital', 'analyst', 'advanced', 'analysis', 'optimisation', 'software', 'development', 'design', 'automation', 'vb.net', 'vba', 'scripting', 'bim',  'software', 'service', 'saas', 'cloud']
                self.categories = [('Business leader', businessleader_words), ('Technical leader', technicalleader_words), ('Digital leader', digitalleader_words), ('Digital designer', digitaldesigner_words), ('Digital analyst', digitalanalyst_words)]
        else:
            self.categories = categories
        # Initialize fdist.
        if fdist == 'default':
            self.fdist = self._getAllInfo()
        else:
            self.fdist = fdist


    def _getInfo(self, name):
        """
        Obtain the information corresponding to the name.

        :param name: the considered name.
        :return: a list of words extracted from the description.
        """
        # Get to location.
        filename = self.folder_name_info + '\\' + name + '\\' + name + '_.txt'
        # Read content of file.
        with open(filename, 'r') as file:
            # Obtain card and bio.
            file_content = file.read()
            file_card = file_content.split('CARD:\n')[1].split('BIO:\n')[0]
            file_bio = file_content.split('BIO:\n')[1]
        # From card, extract name, job, office.
        people_name = file_card.split('\n')[0]
        people_job = file_card.split('\n')[1]
        people_office = file_card.split('\n')[2]
        # From bio, extract different categories and content for each category.
        bio_info = [(content.split('\n')[0], content.split('\n', 1)[-1]) for content in file_bio.split('\n\n')]
        # Define interesting words.
        interesting_words_individual = []
        # Add job description.
        if len(people_job.split(',')) == 2:
            for word in people_job.casefold().split(',')[0].split(' '):
                interesting_words_individual.append(word.casefold())
        # Add content
        for (categories, content) in bio_info:
            for word in re.split('\W+', content):
                interesting_words_individual.append(word.casefold())
        # Remove useless words.
        interesting_words_individual = list(filter(lambda a: a != '', interesting_words_individual))
        return interesting_words_individual


    def _getAllInfo(self):
        """
        Obtain all the information of all persons in the database. We filter
        it using a stopwords list.

        :return: A dictionary correspondig to the frequency distribution of the
        words in all profiles.
        """
        # Define stopwords for the filtering of information
        stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'also']

        # Extract info from all files.
        people_info = []
        # Browse all names.
        for name in os.listdir(self.folder_name_info):
            filename = self.folder_name_info + '\\' + name + '\\' + name + '_.txt'
            # Read content of file.
            with open(filename, 'r') as file:
                # Obtain card and bio.
                file_content = file.read()
                file_card = file_content.split('CARD:\n')[1].split('BIO:\n')[0]
                file_bio = file_content.split('BIO:\n')[1]
                # From card, extract name, job, office.
                people_name = file_card.split('\n')[0]
                people_job = file_card.split('\n')[1]
                people_office = file_card.split('\n')[2]
                # From bio, extract different categories and content for each category.
                bio_info = [(content.split('\n')[0], content.split('\n', 1)[-1]) for content in file_bio.split('\n\n')]
                # Update info file.
                people_info.append((people_name, people_job, people_office, bio_info))

        # Get list of interesting words for each person.
        interesting_words = []
        for (people_name, people_job, people_office, bio_info) in people_info:
            interesting_words_individual = []
            # Add job description.
            if len(people_job.split(',')) == 2:
                for word in people_job.casefold().split(',')[0].split(' '):
                    interesting_words_individual.append(word.casefold())
            # Add content
            for (categories, content) in bio_info:
                for word in re.split('\W+', content):
                    interesting_words_individual.append(word.casefold())
            # Remove useless words.
            interesting_words_individual = list(filter(lambda a: a != '', interesting_words_individual))
            # Append result
            for word in interesting_words_individual:
                interesting_words.append(word)
        # Compute freuency distribution.
        fdist = nltk.FreqDist([word for word in interesting_words if not word in stopwords])

        # Return result.
        return fdist


    def _bagOfWords(self, reference_words, text):
        """
        Computes the 'bag of words' feature of the text in relation to the reference words.

        :param reference_words: A list [(word, number_of_occurence)] corresponding to the different
        reference words, along with their total number of occurence in all descriptions.
        :param text: A list [word] of words in which to search.
        :return: The list [n1, n2, n3, ...] with ni being the number of occurences of the ith reference word in the text
        divided by the total number of occurences of this word in the total environment.
        """
        return [len([word for word in text if word == reference_words[i][0]]) /reference_words[i][1] for i in range(len(reference_words))]


    def _computeProfileFromText(self, text):
        """
        Computes in which profile to classify the text.

        :param text: List of words, corresponding to the text to analyse.
        :return: The string representation of the found profile.
        """
        # Count occurence of each word of each categories in global text.
        new_categories = []
        for (profile, category) in self.categories:
            new_category = []
            for word in category:
                occurence = self.fdist[word]
                if occurence != 0:
                    new_category.append((word, occurence))
            new_categories.append((profile, new_category))
        categories = new_categories

        # For each category, compute bag of word and cumulative sum / length of category.
        scores = []
        for (profile, category) in categories:
            scores.append((profile, sum(self._bagOfWords(category, text)) / len(category)))
        best_score = max(scores, key = lambda x : x[1])
        if best_score[1] > 0:
            return best_score[0]
        else:
            return 'Unable to match description with profile'

    def computeProfile(self, name):
        """
        Computes in which profile to classify the person.
        For that, we look into its description.

        :param name: The name of person.
        :return: The string representation of the found profile.
        """
        try:
            personal_info = self._getInfo(name)
            profile = self._computeProfileFromText(personal_info)
        except Exception as e:
            profile = 'Unable to match description with profile'
        return profile



if __name__ == '__main__':
    # Define face comparator.
    face_comparator = facialRecognition.faceComparator()
    # Define recommender system.
    recommender = basicRecommender(join('Database', 'London_info'))

    # Name of the folder from which we extract the data.
    images_path = join('Database', 'London_images')
    infos_path = join('Database', 'London_info')
    # Name of the file into which we store data.
    file_name = join('Database', 'file')

    # Initialize counters.
    countValidFaces = 0
    countProfiles = {}
    print('Loading faces from folder ' + images_path)

    # Get list of all names.
    list_names = os.listdir(images_path)
    # Initialize result array.
    table_faces = []
    # Browse each name. We use tqdm to make a beautiful progress bar.
    for name in tqdm(list_names, desc = 'Progress', leave = False):
        # Compute profile.
        profile = recommender.computeProfile(name)
        # Get list of images corresponding to name.
        list_images = os.listdir(images_path + '\\' + name)
        # Browse the images. We only keep valid images, i.e. containg one and only one face.
        for image in tqdm(list_images, desc = name, leave = False):
            # Compute encodings.
            encodings = face_comparator.face_encodings(face_comparator.load_image_file(images_path + '\\' + name + '\\' + image))
            if len(encodings) == 1:
                # The image contains only one face, we append it to the data.
                table_faces.append((encodings[0], name, join(images_path, name, image), profile))
                # Actualize counters.
                countValidFaces += 1
                try:
                    countProfiles[profile] += 1
                except:
                    countProfiles[profile] = 1

            else:
                # The image is not valid, we delete the directory.
                shutil.rmtree(images_path + '\\' + name)
    #Save the resulting data.
    np.save(file_name, table_faces)

    # Print results.
    print('Loaded ' + str(countValidFaces) + ' valid faces.')
    for profile, occurences in countProfiles.items():
        print('\tOccurences for ' + profile + ' profile: ' + str(occurences))
