stichometry

## class Sticho (parameters)

**`lang`** `:`      ``'cs', 'de', 'es' ...``  
Language to process (subfolder in "pickle" folder)  

**`method`** `:`     ``'sticho', 'word', 'lemma', '3gram_t' ...``  

   Which data to use for attribution (file in pickle > lang folder)


## class Sticho (methods)

#### Data filtering

**`reduce_features`**  
```
   filters:     conditions to filter features (format accepted by pandas .query method)
                default: None
```
