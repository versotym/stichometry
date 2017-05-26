

## class Sticho (parameters)
<pre>
<b>lang</b> :      'cs', 'de', 'es' ... 
  Language to process (subfolder in "pickle" folder)  

<b>method</b> :     'sticho', 'word', 'lemma', '3gram_t' ...  
  Which data to use for attribution (file in pickle > lang folder)
</pre>

## class Sticho (methods)
#### Data filtering
<pre>
<b>reduce_features</b>
  filters:     conditions to filter features (format accepted by pandas .query method)
               default: None


</pre>
