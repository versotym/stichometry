

## Parameters
<pre>
<b>lang</b> :      <i>'cs', 'de', 'es' ...</i> 
  Language to process (subfolder in "pickle" folder)  

<b>method</b> :     <i>'sticho', 'word', 'lemma', '3gram_t' ...</i>
  Which data to use for attribution (file in pickle > lang folder)
</pre>

## Methods
#### Data filtering
<pre>
<b>reduce_features(filters)</b>
Only for method == 'sticho'
Filter features (columns) which should be used for attribution. 
E.g. drop all statistics on rhyme, or leave on stress profile.
  filters:     conditions to filter features (format accepted by pandas .query method)
               default: None    

<b>mfi(n)</b>
Only for method != 'sticho'
Select how many most frequented items (words, lemmata, n-grams) will be analyzed.
  n:           <i>int</i>
               number of mfi
               default: 500    

<b>reduce_sets(filters, n_min, remove_singles)</b>
Filter datasets (rows) according to specified conditions.
  filters:         conditions to filter datasets (format accepted by pandas .query method)
                   default: None
  n_min:           <i>int</i>
                   minimum number of all features to keep dataset
                   default: 0
  remove_singles:  <i>boolean</i>
                   whether to drop datasets author of which is not author of any other dataset
                   default: True

</pre>

#### Normalization
<pre>
<b>zscores()</b>
Normalize data to z-scores across datasets.
</pre>

#### Attribution
<pre>
<b>nearest_neighbour()</b>
Classification by nearest neighbour (various distance metrics)

<b>svm(multiclass, **kwargs)</b>
Classification by support vector machine
  multiclass:      <i>boolean</i>
                   whether to perform multiclass or binary classification
                   when 'True' each dataset is assigned to one author
                   when 'False' on-vs.-rest. classifier is trained for every author resulting in:
                      (a) assigning author to the dataset if precisely one classifier 
                          gives other decision than 'rest'
                      (b) "I don't know" answer in other cases
                   default: True
  **kwargs:        Parameters for sklearn.svm.SVC (e.g. kernel, gamma...)
  
<b>random_forest(multiclass, **kwargs)</b>  
Classification by random forest
  multiclass:      <i>boolean</i>
                   whether to perform multiclass or binary classification
                   when 'True' each dataset is assigned to one author
                   when 'False' on-vs.-rest. classifier is trained for every author resulting in:
                      (a) assigning author to the dataset if precisely one classifier 
                          gives other decision than 'rest'
                      (b) "I don't know" answer in other cases
                   default: True
  **kwargs:        Parameters for sklearn.ensemble.randomForestClassifier 
                   (e.g. n_estimators, class_weight...)
</pre>

#### Evaluation
<pre>
<b>evaluate()</b>
Print evaluation of particulars methods that were applied

<b>dendrograms()</b>
Plot dendrograms (only if nearest_neighbour has been applied)

<b>complete_results(pickle, filename)</b>
Returns dictionary with complete results
  pickle:          <i>boolean</i>
                   whether to pickle dict into a file (stored in 'pickle' folder)
                   default: True
  filename:        specifies the name of a pickled file
                   default: method name (e.g. sticho, word...)
