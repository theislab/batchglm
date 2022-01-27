
# Fast and scalable fitting of over-determined generalized-linear models (GLMs)

batchglm was developed in the context of [diffxpy](https://github.com/theislab/diffxpy) to allow fast model fitting for differential expression analysis for single-cell RNA-seq data. However, one can use batchglm or its concepts in other scenarios where over-determined GLMs are encountered.

```
pip install -r requirements.txt
```

To run unit tests:

```
pip install -e .
python -m unittest
```