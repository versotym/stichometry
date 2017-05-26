

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
  filters:     conditions to filter features (format accepted by pandas .query method)
               default: None  
  
<b>reduce_sets(filters, n_min, remove_singles)</b>
  filters:     conditions to filter datasets (format accepted by pandas .query method)
               default: None
  n_min:       <i>int</i>
               minimum number of all features to keep dataset
               default: 0

</pre>
