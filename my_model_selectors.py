import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, n_iter=1000, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.n_iter = n_iter
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def train_model(self, num_states, x_train, x_len):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=self.n_iter,
                                    random_state=self.random_state, verbose=False).fit(x_train, x_len)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except Exception as e:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
                print("Exception: {}".format(e))
            return None

    def base_model(self, num_states):
        return self.train_model(num_states, self.X, self.lengths)

    def get_data_shape(self):
        return self.X.shape

    def score_model(self, model, x, x_len):
        try:
            return model.score(x, x_len)
        except Exception as e:
            if self.verbose:
                print("failure scoring model for {}".format(self.this_word))
                print("Exception: {}".format(e))
            return float("-inf")

    def base_score(self, model):
        return self.score_model(model, self.X, self.lengths)


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # implement model selection based on BIC scores
        n, features = self.get_data_shape()
        logN = np.log(n)
        components = range(self.min_n_components, self.max_n_components + 1)
        models = map(self.base_model, components)
        scores = map(self.base_score, models)
        n_params = map(lambda n_com: n_com * (n_com - 1) + 2 * features * n_com, components)
        bic_score = map(lambda log_params: -2 * log_params[0] + log_params[1] * logN, zip(scores, n_params))

        best_score, best_n = max(zip(bic_score, components))
        return self.base_model(best_n)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def model_for_word(self, n_components, word):
        x, x_len = self.hwords[word]
        model = self.train_model(n_components, x, x_len)
        score = self.score_model(model, x, x_len)
        return model, score

    def get_all_scored_models(self):
        component_models = {}
        # Train and score individial word models and components combinations
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            word_models = {}
            for word in self.words.keys():
                model, score = self.model_for_word(n_components, word)
                if model:
                    word_models[word] = model, score

            component_models[n_components] = word_models
        return component_models

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score, best_n_components = float("-inf"), None
        for n_components, word_models in self.get_all_scored_models().items():
            if self.this_word not in word_models:
                continue
            other_words = [word for word in word_models.keys() if word != self.this_word]
            avg = np.mean([word_models[word][1] for word in other_words])
            dic = word_models[self.this_word][1] - avg
            if dic > best_score:
                best_score, best_n_components = dic, n_components

        return self.base_model(best_n_components if best_n_components else 3)


class AllData(object):
    def split(self, data):
        return [(range(len(data)), range(len(data)))]


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        def get_train_test_ds(n_splits):
            split_method = (KFold(random_state=self.random_state, n_splits=n_splits)
                            if len(self.sequences) >= 3 else AllData())

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                x_train, x_train_len = combine_sequences(cv_train_idx, self.sequences)
                x_test, x_test_len = combine_sequences(cv_test_idx, self.sequences)
                yield x_train, x_train_len, x_test, x_test_len

        def get_score(n_components, data):
            x_train, x_train_len, x_test, x_test_len = data
            model = self.train_model(n_components, x_train, x_train_len)
            score = self.score_model(model, x_test, x_test_len)

            return score

        def model_with_n_components(n_components):
            scores = list(get_score(n_components, data) for data in get_train_test_ds(3))
            if len(scores) != n_components:
                return float("-inf"), n_components
            else:
                return np.mean(scores), n_components

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        res = list(model_with_n_components(n) for n in range(self.min_n_components, self.max_n_components + 1))
        best_score, best_n = max(res)

        return self.base_model(best_n)
