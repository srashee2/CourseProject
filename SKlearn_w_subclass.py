#!/usr/bin/env python
# coding: utf-8

# # Import essential libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer
import pyLDAvis.sklearn


# In[2]:


def lemma_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in text]


# In[3]:


main_df = pd.read_csv('NYT2000_1.csv', usecols=['Body', 'Publication Day Of Month', 'Publication Month', 'Publication Year'])
temp_df = pd.read_csv('NYT2000_2.csv', usecols=['Body', 'Publication Day Of Month', 'Publication Month', 'Publication Year'])
main_df = pd.concat([main_df,temp_df], ignore_index=True)


# In[4]:


# Remove NaN values, lowercase contents of Body column, filters for bush and gore and resets the index
print(main_df.shape)
main_df.dropna(subset=['Body'], inplace=True)
print(main_df.shape)

main_df['Body'] = main_df['Body'].str.lower()
main_df = main_df[main_df['Body'].str.contains('gore|bush')]
main_df = main_df.reset_index(drop=True)


# In[5]:


# Create a single date column from day, month and year columns
main_df['Date'] = pd.to_datetime(
    main_df['Publication Year'] * 10000 + main_df['Publication Month'] * 100 + main_df['Publication Day Of Month'],
    format='%Y%m%d')
main_df.drop(['Publication Year', 'Publication Month', 'Publication Day Of Month'], axis=1, inplace=True)
print(main_df.shape)


# In[6]:


# Remove unnecessary symbols, numbers, words less than 3 characters and apply lemmatizer
main_df['Body'].replace([r'[,\.!?]', r'\d+', r'\b(\w{1,2})\b'], '', inplace=True, regex=True)
main_df['Body'].apply(lemma_text)
main_df['Body'] = main_df['Body'].str.replace('said', '', regex=False)
print(main_df['Body'].head(10))


# In[7]:


# Generate doc-term matrix
cv = CountVectorizer(stop_words='english', max_df=3500)
ft_cv = cv.fit_transform(main_df['Body'])
vocabulary = cv.get_feature_names()

doc_term_matrix = pd.DataFrame(ft_cv.toarray(), columns=vocabulary)
print(doc_term_matrix.shape)


# In[25]:





# In[27]:


from sklearn.utils import check_random_state
from sklearn.decomposition._online_lda_fast import _dirichlet_expectation_2d
class PTWGuidedLatentDirichletAllocation(LatentDirichletAllocation):

    def __init__(self, n_components=10, doc_topic_prior=None, topic_word_prior=None, learning_method='batch', learning_decay=0.7, learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=-1, total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=None, verbose=0, random_state=None, n_topics=None, ptws=None):
        super(PTWGuidedLatentDirichletAllocation, self).__init__(n_components, doc_topic_prior, topic_word_prior, learning_method, learning_decay, learning_offset, max_iter, batch_size, evaluate_every, total_samples, perp_tol, mean_change_tol, max_doc_update_iter, n_jobs, verbose, random_state, n_topics)
        self.ptws = ptws

    def _init_latent_vars(self, n_features):
        """Initialize latent variables."""

        self.random_state_ = check_random_state(self.random_state)
        self.n_batch_iter_ = 1
        self.n_iter_ = 0

        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = 1. / self.n_topics
        else:
            self.doc_topic_prior_ = self.doc_topic_prior

        if self.topic_word_prior is None:
            self.topic_word_prior_ = 1. / self.n_topics
        else:
            self.topic_word_prior_ = self.topic_word_prior

        init_gamma = 100.
        init_var = 1. / init_gamma
        # In the literature, this is called `lambda`
        self.components_ = self.random_state_.gamma(
            init_gamma, init_var, (self.n_topics, n_features))

        # Transform topic values in matrix for prior topic words
        if self.ptws is not None:
            for ptw in self.ptws:
                word_index = ptw[0]
                word_topic_values = ptw[1]
                self.components_[:, word_index] *= word_topic_values

        # In the literature, this is `exp(E[log(beta)])`
        self.exp_dirichlet_component_ = np.exp(
            _dirichlet_expectation_2d(self.components_))


# In[31]:


# Fit LDA model to doc-term matrix
k = 15

lda = Lda(n_components=k)

lda.fit(ft_cv)

print('log likelihood score, 15 topics: ' + str(lda.score(ft_cv)))
pyLDAvis.enable_notebook()
p = pyLDAvis.sklearn.prepare(lda, ft_cv, cv)
pyLDAvis.display(p)


# In[16]:


# Generate doc-topic matrix
lda_out = lda.transform(ft_cv)
doc_topic_matrix = pd.DataFrame(lda_out)
doc_topic_matrix['Date'] = main_df['Date']
print(doc_topic_matrix.shape)

aggregator = {i: 'sum' for i in range(k)}
coverage_curve = doc_topic_matrix.groupby(['Date']).agg(aggregator)
print(coverage_curve.shape)
print(coverage_curve.head(10))

plt.rcParams['figure.figsize'] = [16, 10]
plt.figure()
coverage_curve.plot()
plt.show()


# In[10]:


#Read the IEM data and Normalize one of the stocks
iem_data = pd.read_excel('IEM2000.xlsx')
iem_data.drop(['Units', '$Volume', 'LowPrice','HighPrice','AvgPrice'], axis=1, inplace=True)

dem_data = iem_data[iem_data['Contract'].str.contains('Dem')]
rep_data = iem_data[iem_data['Contract'].str.contains('Rep')]

dem_data.set_index('Date', inplace=True)
rep_data.set_index('Date', inplace=True)

dem_data['NormLastPrice'] = dem_data['LastPrice'] / (dem_data['LastPrice'] + rep_data['LastPrice'])
dem_data.drop(['LastPrice', 'Contract'], axis=1, inplace=True)

print(dem_data.head())


# In[14]:


combined_data = pd.concat([dem_data, coverage_curve], axis=1, join='inner')
print(combined_data.head())
print(combined_data.tail())
print(pd.date_range(start='2000-05-01',end='2000-11-10').difference(combined_data.index))


# In[12]:


from statsmodels.tsa.stattools import grangercausalitytests
granger_results1 = grangercausalitytests(combined_data[['NormLastPrice',2]],5)
granger_results2 = grangercausalitytests(combined_data[['NormLastPrice',7]],5)
granger_results3 = grangercausalitytests(combined_data[['NormLastPrice',10]],5)
granger_results4 = grangercausalitytests(combined_data[['NormLastPrice',11]],5)
granger_results5 = grangercausalitytests(combined_data[['NormLastPrice',12]],5)


# In[ ]:




