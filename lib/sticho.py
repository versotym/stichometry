import os
import sys
import pandas
import re
import pickle
import dill
import itertools
import shutil
import numpy as np
import pandas as pd
from pprint import pprint
from collections import defaultdict, OrderedDict
from scipy import stats, linalg
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from scipy.spatial.distance import cosine, squareform, pdist
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import rcParams
from pathlib import Path
rcParams.update({'figure.autolayout': True})


class Sticho:

    # =========================================================================
    # CONSTRUCTOR
    # =========================================================================

    def __init__(self, lang='cs', method='sticho'):
        '''Load pickled dataframes:

                      {module, feature, spec1, spec2, type, value}
        == x ==       CATEGORY1   CATEGORY2  CATEGORY3  ...    [STICHO/STYLO]
        {auth, set id}
        SUBCORPUS1    x1,1        x1,2       x1,3       ...
        SUBCORPUS2    x2,1        x2,2       x2,3       ...
        ...           ...         ...        ...        ...

                      {module, feature, spec, spec2, type, value}
        == n ==       CATEGORY1   CATEGORY2  CATEGORY3  ...    [STICHO]
        {auth, set id}
        SUBCORPUS1    n1,1        n1,2       n1,3       ...
        SUBCORPUS2    n2,1        n2,2       n2,3       ...
        ...           ...         ...        ...        ...

        == n ==                                                [STYLO]
        {auth, set id}
        SUBCORPUS1    # of items
        SUBCORPUS2    # of items
        ...
        '''

        directory = str(Path(__file__).parents[1])
        file = os.path.join(directory, 'pickle', lang, method + '.pickle')
        with open(file, 'rb') as handle:
            (self.x, self.n) = pickle.load(handle)

        self.lang = lang
        self.method = method
        self.results = dict()
        self.results_weighted = defaultdict(dict)
        self.metrics = ('euclidean', 'cosine', 'cityblock', 'chebyshev',
                        'correlation', 'canberra', 'argamon')

    # =========================================================================
    # DATA FILTERS
    # =========================================================================

    def reduce_features(self, filters=None):
        '''Apply filters to features (list of expressions)'''

        if not self.method.startswith('sticho'):
            raise Exception('''Method applies to "sticho" sets only.
                            For "stylo" use .mfi()''')
        if filters:
            for f in filters:
                self.x = self.x.T.query(f).T
                self.n = self.n.T.query(f).T

    def mfi(self, n=500):
        '''Reduce stylo data to n most frequented items'''

        if self.method.startswith('sticho'):
            raise Exception('''Method applies to "stylo" sets only.
                            For "sticho" use .reduce_features()''')
        sums = self.x.sum(numeric_only=True)
        sums.name = ('frequency_in', 'entire', 'corpus')
        self.x = self.x.append(sums)
        self.x = self.x.T.sort_values(self.x.index[-1], ascending=False).T
        self.x = self.x.ix[:-1]
        self.mfi = n
        self.x = self.x.iloc[:, 0:n]

    def reduce_sets(self, n_min=0, remove_singles=True,
                    filters=None, target=None):
        '''Apply filters to datasets (list of expressions). Drop sets if
        number of some feature (sticho) or total number of items (stylo)
        is lower than n_min. Drop authors of one set only if
        remove_singles==True
        '''

        if filters:
            for f in filters:
                self.x = self.x.query(f)
                self.n = self.n.query(f)
        if self.method == 'sticho':
            self.x = self.x.loc[(self.n >= n_min).all(axis=1), :]
            self.n = self.n.loc[(self.n >= n_min).all(axis=1), :]
        else:
            #self.x = self.x.loc[(self.n >= n_min), :]
            #self.n = self.n.loc[(self.n >= n_min), :]
            pass
        if remove_singles:
            uniques = list(self.x.index.get_level_values('author')
                           .drop_duplicates(keep=False))
            if target:
                uniques = [a for a in uniques if a not in target]
            self.x = self.x.drop(uniques, level='author')
            self.n = self.n.drop(uniques, level='author')
        if len(self.x.index) <= 1:
            raise Exception('# of sets <= 1 after reducing')

    def autoreduce_features(self, n_min=10):
        '''Drop all features which occur in less than n_min cases 
        in one or more datasets
        '''
        if self.method != 'sticho':
            raise Exception('''Method applies to "sticho" sets only.
                               For "stylo" use .mfi()''')
        self.x = self.x.loc[:, (self.n >= n_min).all(axis=0)]
        self.n = self.x.loc[:, (self.n >= n_min).all(axis=0)]
        if len(self.x.columns) <= 1:
            raise Exception('# of features <= 1 after autoreducing')

    def get_datasets_list(self):
        '''Get list of datasets. This may be further used with another 
        file within method apply_datasets_list()
        '''
        return self.x.index

    def apply_datasets_list(self, list_):
        '''Reduce datasets according to another file'''
        self.x = self.x.loc[list_]
        self.n = self.n.loc[list_]

    # =========================================================================
    # DATA NORMALIZATION
    # =========================================================================

    def zscores(self):
        '''Normalize features across datasets to z-scores'''

        self.x = (self.x - self.x.mean())/self.x.std(ddof=0)
        self.x = self.x.fillna(0)

    # =========================================================================
    # NEAREST NEIGHBOUR
    # =========================================================================

    def nearest_neighbour(self):
        '''Apply different distance metrics'''

        for metric in self.metrics:
            dist_matrix = self._distance_matrix(metric, self.x)
            self.results[metric] = self._distance_metrics_accuracy(dist_matrix)

    def nearest_neighbour_weighted(self):
        '''Apply different distance metrics to particular feature sets
        separately and get the mean value
        '''

        sums = dict()
        features = set(self.x.columns.get_level_values('feature').unique())
        for f in features:
            x_slice = self.x.xs(f, level='feature', drop_level=False, axis=1)
            for metric in self.metrics:
                dist_matrix = self._distance_matrix(metric, x_slice)
                dist_matrix = (dist_matrix - dist_matrix.mean()) / dist_matrix.std(ddof=0)
                dist_matrix = dist_matrix.fillna(0)
                if metric not in sums:
                    sums[metric] = dist_matrix
                else:
                    sums[metric] = sums[metric]+dist_matrix
                np.fill_diagonal(dist_matrix.values, np.nan)
                self.results_weighted[metric][f] = self._distance_metrics_accuracy(dist_matrix)
        for metric in sums:
            np.fill_diagonal(sums[metric].values, np.nan)
            self.results['*'+metric] = self._distance_metrics_accuracy(sums[metric])

    def _distance_matrix(self, metric, data):
        '''-- distance matrix'''

        if metric == 'argamon':
            return self._rotated_delta(data)
        else:
            dist_matrix = pd.DataFrame(data=squareform(pdist(data, metric)),
                                       columns=data.index,
                                       index=data.index)
        np.fill_diagonal(dist_matrix.values, np.nan)
        return dist_matrix

    def _rotated_delta(self, data):
        '''-- calculate rotated (non-axis parallel) delta (Argamon 2009)
        NOT SURE IF THIS IS IMPLEMENTED CORRECTLY
        '''

        cov = data.apply(pd.to_numeric, errors='coerce').cov()
        ev, E = linalg.eigh(cov)
        select = np.array(ev, dtype=bool)
        D_ = np.diag(ev)  # (ev[select])
        E_ = E[:, select]
        D_inv = D_.T

        deltas = pd.DataFrame(index=data.index, columns=data.index)
        for d1, r1 in data.iterrows():
            for d2, r2 in data.iterrows():
                diff = (r1 - r2)
                delta = diff.T.dot(E).dot(D_inv).dot(E.T).dot(diff)
                deltas.at[d1, d2] = delta
                deltas.at[d2, d1] = delta
        np.fill_diagonal(deltas.values, np.nan)
        return deltas

    def _distance_metrics_accuracy(self, dist_matrix):
        '''-- count success rate for distance matrix'''

        match = 0
        for r in dist_matrix.index:
            if r[0] == dist_matrix[r].idxmin()[0]:
                match += 1
        return {'accuracy': match / len(dist_matrix.index),
                'distance_matrix': dist_matrix}

    # =========================================================================
    # MACHINE LEARNING
    # =========================================================================

    def svm(self, multiclass, **kwargs):
        '''Support vector machine learning techniques'''

        model = SVC(**kwargs)
        if multiclass:
            self._machine_learning_multiclass(method='svm', model=model)
        else:
            self._machine_learning_binary(method='svm', model=model)

    def random_forest(self, multiclass, **kwargs):
        '''Random forest learning technique'''
        
        model = RandomForestClassifier(n_jobs=n_jobs,
                                       n_estimators=n_estimators)
        self.random_forest_importances = defaultdict(lambda: defaultdict(int))
        self._machine_learning_multiclass(method='rfo', model=model)
        self._machine_learning_binary(method='rfo', model=model)

    def _machine_learning_multiclass(self, method, model):
        '''-- machine learning with multiple classes'''

        self.results[method+'_multiclass'] = {
            'accuracy': 0,
            'decisions': dict(),
        }

        for row in self.x.iterrows():
            index, data = row
            target_author = index[0]
            target_vector = [data.tolist()]
            training_authors = self.x.drop(index).index.get_level_values('author').tolist()
            training_vectors = self.x.drop(index).values.tolist()
            model.fit(training_vectors, training_authors)
            predicted = model.predict(target_vector)
            print (target_author, ' => ' ,predicted)
            if target_author == predicted:
                self.results[method+'_multiclass']['accuracy'] += 1
            self.results[method+'_multiclass']['decisions'][index] = predicted[0]
            if method == 'rfo':
                for i, f in enumerate(model.feature_importances_):
                    self.random_forest_importances['multiclass']['_'.join(self.x.columns[i][0:3])] += f
                    self.random_forest_importances['multiclass_count']['_'.join(self.x.columns[i][0:3])] += 1

        self.results[method+'_multiclass']['accuracy'] /= len(self.x.index)

    def _machine_learning_binary(self, method, model):
        '''-- machine learning with binary classification'''

        self.results[method + '_binary'] = {
            'accuracy': defaultdict(int),
            'decisions': defaultdict(set),
        }

        authors = set(self.x.index.get_level_values('author').unique())

        for row in self.x.iterrows():
            index, data = row
            target_author = index[0]
            target_vector = [data.tolist()]
            training_vectors = self.x.drop(index).values.tolist()

            for a in authors:
                training_authors = self.x.drop(index).index.get_level_values('author').tolist()
                training_authors = ['non' if x != a else x for x in training_authors]
                if len(set(training_authors)) <= 1:
                    continue
                model.fit(training_vectors, training_authors)
                predicted = model.predict(target_vector)
                if predicted[0] != "non":
                    self.results[method+'_binary']['decisions'][index].add(predicted[0])

            if len(self.results[method+'_binary']['decisions'][index]) == 0:
                self.results[method+'_binary']['accuracy']['dunno'] += 1
            elif len(self.results[method+'_binary']['decisions'][index]) == 1:
                if target_author in self.results[method+'_binary']['decisions'][index]:
                    self.results[method+'_binary']['accuracy']['succ'] += 1
                else:
                    self.results[method+'_binary']['accuracy']['fail'] += 1
            else:
                self.results[method+'_binary']['accuracy']['dunno'] += 1

        self.results[method+'_binary']['accuracy']['succ'] /= len(self.x.index)
        self.results[method+'_binary']['accuracy']['fail'] /= len(self.x.index)
        self.results[method+'_binary']['accuracy']['dunno'] /= len(self.x.index)

    def random_forest_fi(self):
        '''Features importances in random forest'''

        means = dict()
        for f in self.random_forest_importances['multiclass']:
            means[f] = (
                self.random_forest_importances['multiclass'][f]
                / self.random_forest_importances['multiclass_count'][f]
            )
        return sorted(means.items(), key=lambda x: x[1], reverse=True)

    # =========================================================================
    # EVALUATION
    # =========================================================================

    def evaluate(self):
        '''Evaluation of particular distant metrics and learning techniques'''

        print("__________________________________")
        print("                        EVALUATION\n")
        self._print_accuracy()
        print("__________________________________\n")

    def _general_report(self):
        '''-- general info on dataframe '''
        r = OrderedDict()
        r['lang'] = self.lang
        r['method'] = self.method
        r['authors'] = len(self.x.index.get_level_values('author').unique())
        r['datasets'] = len(self.x.index)
        r['features'] = len(self.x.columns)
        r['baseline'] = self._random_baseline()
        return r

    def _random_baseline(self):
        '''--random baseline of attribution process'''

        rb = 0
        n_datasets = len(self.x.index)
        authors = set(self.x.index.get_level_values('author').unique())
        for a in authors:
            n = len(self.x.xs(a, level='author', drop_level=False, axis=0).index)
            rb += ((n-1)/n_datasets)*(n/n_datasets)
        return rb

    def _print_accuracy(self):
        '''-- accuracy table'''

        print ('metric   success             fail')
        print ('----------------------------------')

        for metric in sorted(self.results):
            if type(self.results[metric]['accuracy']) == float:
                print('{0}    {1:.4f}  {2}{3}'.format(
                    metric[:5].upper(),
                    self.results[metric]['accuracy'],
                    '▮' * int(10*self.results[metric]['accuracy']),
                    '▯' * (10-int(10*self.results[metric]['accuracy'])), ))
            else:
                succ_bar = int(10*self.results[metric]['accuracy']['succ'])
                fail_bar = int(10*self.results[metric]['accuracy']['fail'])
                print('{0}    {1:.4f}  {2}{3}{4}  {5:.4f}'.format(
                    metric[:5].upper(),
                    self.results[metric]['accuracy']['succ'],
                    '▮' * succ_bar,
                    ' ' * (10 - succ_bar - fail_bar),
                    '▯' * fail_bar,
                    self.results[metric]['accuracy']['fail'], ), flush=True)

    # =========================================================================
    # COMPLETE RESULTS
    # =========================================================================

    def complete_results(self, pickle=True, filename=None):
        '''Produce dict with complete results that may be stored'''

        cr = {
            'head': self._general_report(),
            'results': self.results,
            'nearest_neighbour_subsets': self.results_weighted
        }

        if hasattr(self, 'random_forest_importances'):
            cr['random_forest_importances'] = self.random_forest_fi()
        if pickle:
            self._pickle_results(filename, cr)
        return cr

    def _pickle_results(self, filename, cr):
        '''-- pickle and store complete results'''

        if not filename:
            filename = self.method
        directory = self._results_directory()
        file = os.path.join(directory, filename+'.pickle')
        with open(file, 'wb') as handle:
            pickle.dump(cr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _results_directory(self):
        '''-- specify folder to store pickled results
        and create it if it does not exists
        '''

        parent = str(Path(__file__).parents[1])
        if not os.path.exists(os.path.join(parent, 'results')):
            os.makedirs(os.path.join(parent, 'results'))
        if not os.path.exists(os.path.join(parent, 'results', self.lang)):
            os.makedirs(os.path.join(parent, 'results', self.lang))
        return os.path.join(parent, 'results', self.lang)

    # =========================================================================
    # DENDROGRAMS
    # =========================================================================

    def dendrograms(self):
        '''Create matrix for dendrogram'''

        print('Plotting dendrograms...')    
        directory = self._dendrograms_directory()
        for method in ['average', 'complete', 'single', 'ward']:
            for metric in ['euclidean', 'cosine', 'cityblock', 'correlation']:
                if method == 'ward' and metric != 'euclidean':
                    continue
                linkage_matrix = linkage(self.x, method=method, metric=metric)
                self._plotting(method, metric, linkage_matrix, directory)

    def _dendrograms_directory(self):
        '''-- specify folder to store dendrograms
        and create it if it does not exists
        '''

        parent = str(Path(__file__).parents[1])
        if not os.path.exists(os.path.join(parent, 'img')):
            os.makedirs(os.path.join(parent, 'img'))
        if not os.path.exists(os.path.join(parent, 'img', self.lang)):
            os.makedirs(os.path.join(parent, 'img', self.lang))
        return os.path.join(parent, 'img', self.lang)

    def _plotting(self, method, metric, linkage_matrix, directory):
        '''-- plot dendrograms'''

        file = os.path.join(directory, self.method + '_' + metric + '_' + method + '.png')
        (leaves, colors) = self._dendrogram_labels()
        plt.ioff()
        fig = plt.figure(figsize=(10, int(len(leaves)/10)+4))
        plt.title(self.lang+' :: '+self.method+" :: "+metric+" ("+method+") :: :: :: :: :: "\
            "aut.: " + str(len(self.x.index.get_level_values('author').unique())) + " :: "\
            "sets: " + str(len(self.x.index)) + " :: "\
            "feat.: " + str(len(self.x.columns)), 
        )
        ddata = dendrogram(linkage_matrix,
                           color_threshold=0.00001,
                           labels=leaves,
                           orientation='right',
                           leaf_font_size=8
                           )
        ax = plt.gca()
        xlbls = ax.get_ymajorticklabels()
        for lbl in xlbls:
            lbl.set_color(colors[lbl.get_text()])
        plt.savefig(file)
        plt.close(fig)

    def _dendrogram_labels(self):
        '''-- add labels to dendrogram'''

        palette = ("r", "g", "b", "m", "k",
                   "Olive", "SaddleBrown", "CadetBlue", "DarkGreen", "Brown")
        authors = self.x.index.get_level_values('author').unique()
        (leaves, colors) = (list(), dict())
        for a in list(self.x.index):
            leaves.append(' :: '.join(a[0:2]))
            colors[' :: '.join(a[0:2])] = palette[authors.get_loc(a[0]) % 10]
        return (leaves, colors)

    # =========================================================================
    # CURRENT DATA OVERVIEW
    # =========================================================================

    def overview(self, axis='datasets', pause=True):
        '''Print all the features|datasets currently in dataframe'''

        print("__________________________________\n")
        for r, val in self._general_report().items():
            print ('{0}:'.format(r).ljust(10), val)

        if (axis == 'datasets'):
            print("_ _________________________________")
            print("                 DATASETS ANALYZED\n")
            for i, c in enumerate(list(self.x.index)):
                print('\033[33m{0}\033[0m  {1}'.format(i+1, c))
        elif (axis == 'features'):
            if self.method != 'sticho':
                raise Exception('For "stylo" sets use .features_detail')
            print("__________________________________")
            print("               CATEGORIES ANALYZED\n")
            cats = defaultdict(int)
            for r in list(self.x.columns):
                cats[r[0] + ' ' + r[1]] += 1
            for i, c in enumerate(sorted(cats)):
                print('\033[33m{0}\033[0m  {1} => {2}'.format(i+1, c, cats[c]))
        elif (axis == 'features_detail'):
            print("__________________________________")
            print("               CATEGORIES ANALYZED (DETAIL)\n")
            for i, r in enumerate(list(self.x.columns)):
                print('\033[33m{0}\033[0m  {1}'.format(i+1, r))
                print("__________________________________")
        if pause:
            print("           Press enter to continue")
            input()
        else:
            print()

    def inspect_datasets(self, author=None, deep=True):
        '''Print detailed info on datasets of specified author'''
        for row in self.n.iterrows():
            index, data = row
            if author and not index[0].startswith(author):
                continue
            output = defaultdict(int)
            subfeatures = defaultdict(int)
            for col in data.iteritems():
                feature, val = col
                if deep:
                    output[feature[0:3]] = val          
                    subfeatures[feature[0:3]] += 1  
                else:
                    output[feature[2]] = val
                    subfeatures[feature[2]] += 1  
            print("__________________________________")
            print(index)
            print("__________________________________")
            for f in sorted(output):
                print('{0} {1} = {2}'.format(f, subfeatures[f], output[f]))
            input()
