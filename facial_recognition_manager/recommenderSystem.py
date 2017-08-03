# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 19:05:06 2017

@author: Lucas
"""

"""
The purpose of this module is to implement an algorithm for the recommendation.
We want it to implement 1 function :
    computeProfile(name),
which assigns a category to the given name.
The different categories we aim at are :
    - Business leader.
    - Techical leader.
    - Digital leader.
    - Digital designer.
    - Digital analyst.
If the algorithm does not fing a category, it returns the string :
    'Unable to match description with profile'.
"""

###############################################################################
# Imports.
###############################################################################


###############################################################################
# Main content of the module.
###############################################################################


class basicRecommender:
    """
    A class for basic computation for recommendations.
    """
    def __init__(self, database, fdist = 'default', categories = 'default'):
        """
        Initialization of the class.

        :param fdist: The frequency distribution of words in the whole description.
        :param categories: The different categories in which to classify people, along with corresponding keywords.
        """
        # Initialize database.
        self.database = database
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
            self.fdist = database.getAllInfo()
        else:
            self.fdist = fdist


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
        return self._computeProfileFromText(self.database.getInfo(name))
