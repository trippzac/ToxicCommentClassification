# Goal of the Project
This project is based on a previous Kaggle competition. Six types of toxicity of comments were given as follows:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

The goal of the project is given an online comment, determine the probability for each category above that the comment falls into said category. Further information on this competition, including the data used for testing and training, can be found at the [Kaggle competition website](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/).

## Status of the Project
This project is currently being worked on and will be updated regularly for the near future. All Python notebooks so far have been made using Google Colab.

# Data Processing and Cleaning
Beyond the initial work of combining the test set with its labels and getting rid of comments that do not have a scoring associated to them, I implemented my own method to clean the text data. This involves using the stop words and WordNetLemmatizer of the Natural Language Toolkit (NLTK), stripping words of undesirable characters, and getting rid of words of length less than 3. Additionally, the current version of the model uses TfidfVectorizer from scikit-learn with a maximum of 100,000 features in order to represent each text. Two versions of the current logistic model have been tried, one using n-grams of length greater than 1 and another using only 1-grams.

# Models
A handful of basic versions of the model have been tried. The first iteration of the project utilized LogisticRegression, LinearSVC, GradientBoostingClassifier, and RandomForestClassifier from scikit-learn. It was expected from the beginning that logistic regression or linear SVC would work best given that multiple resources suggested that NLP classification problems similar to this one are linearly separable. This turned out to be the case. The data was run on the vectorized data and on data in which the dimension was further reduced using TruncatedSVD. We display the results using ROC AUC score (the metric used for the Kaggle competition) rounded to four digits. Most versions of the data were not run using gradient boosting or random forests due to the lengthy runtimes needed. 

 |                             | LogisticRegression | LinearSVC | GradientBoostingClassifier | RandomForestClassifier |
 | --------------------------- | ------------------ | --------- | -------------------------- | ---------------------- |
 | No dimensionality reduction | 0.9083 | 0.8523 | -- | -- |
 | Dimensionality reduction with 3.2% of variance explained | -- | -- | 0.6130 | 0.5020 |
 | Dimensionality reduction with 12.2% of variance explained | 0.8679 | 0.8634 | -- | -- |
 | Dimensionality reduction with 37.9% of variance explained | 0.8999 | 0.8930 | -- | -- |
 
Due to increasingly significant runtimes, the dimensionality reduction was unable to explain much of the variance even for logistic regression and linear SVC. TdidfVectorizer has the benefit that the vectors are stored as sparse vectors. However, this was still insufficient to run gradient boosting or random forest in a reasonable time frame. It should be noted that these are the scores when the models determined *predictions* of the data instead of *probabilities*, hence the significantly lower scores than most given for the competition.

## Current Models
Currently, the best results have been found using logistic regression and hyperparameter tuning on the regularization constant C. One model uses only 1-grams to determine the word embedding, whereas the other uses n-grams for n = 1, 2, 3. Initially, there was some minor improvement over previous models to a score of 0.9161 using just 1-grams. However, a major issue was fixed in this version. The competition was being scored with a vector of probabilities, not predictions. Once this was rectified, the model using only 1-grams obtained a score of 0.9752, while the model using 2-grams and 3-grams as well obtained a score of 0.9747.

# Next Steps
1. There appears to be some conceptual similarity between some of the words often found in misclassified comments. For example, for comments that are misclassified with regards to if they are a threat or not, words such as 'article', 'page', 'wikipedia', 'source', 'edit', and 'section' appear frequently. This likely means that the vectorizer is associating these related words with other threatening words. A way to improve this will be to use pre-trained databases for vectorization.
2. Create a deep learning model such as BERT. My goal from the beginning was to eventually create a deep learning model that is useful for sentiment analysis, and this is likely the correct next step.

# References
* [Scikit-learn: Machine Learning in Python](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
* [Kaggle Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/)
* [Natural Language Toolkit](https://www.nltk.org/)
* [N-grams](https://towardsdatascience.com/understanding-word-n-grams-and-n-gram-probability-in-natural-language-processing-9d9eef0fa058)
