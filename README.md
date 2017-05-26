

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
