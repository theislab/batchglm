Implementation of models using Tensorflow
====
This module contains all model estimators depending on Tensorflow.


Template to implement a new model estimator:
----
First, set up a parameter definition defining all model parameters together with the corresponding dimensions:
```python
PARAMS = {
    "param_1": ("samples", "variables"),
    "param_2": ("variables",),
    ...
}
```
All equally-named dimensions have to be of the same size.

Create a Tensorflow model with all necessary parameters:
```python
from impl.tf.base import TFEstimatorGraph

class EstimatorGraph(TFEstimatorGraph):
    def __init__(self, graph):
        TFEstimatorGraph.__init__(self, graph)
        # required by TFEstimatorGraph
        self.global_step = tf.train.get_or_create_global_step()
        self.init_op = ...
        self.loss = ...
        self.train_op = ...
        # parameters:
        self.param_1 = ...
        self.param_2 = ...
        
```
Now create the actual Estimator for the given model:
```python
from models.<some_model> import AbstractEstimator
from impl.tf.base import MonitoredTFEstimator

class SomeEstimator(AbstractEstimator, MonitoredTFEstimator, metaclass=abc.ABCMeta):
    model: EstimatorGraph
    
    # Set up a PARAMS property returning the previously created parameter definition:
    #   This property is used among other things for exporting data to NetCDF-format.
    @property
    def PARAMS(cls) -> dict:
        return PARAMS
    
    def __init__(self, input_data, model=None):
        if model is None:
            tf.reset_default_graph()
            # create model
            model = EstimatorGraph(graph=tf.get_default_graph())
        
        MonitoredTFEstimator.__init__(self, input_data, model)
    
    # The scaffold provides some information about the model graph to the training session.
    #   It is possible to add additional capabilities like a summary_op which writes summaries for TensorBoard 
    tf1
    def _scaffold(self):
        with self.model.graph.as_default():
            scaffold = tf.train.Scaffold(
                init_op=self.model.init_op,
                summary_op=self.model.merged_summary,
                saver=self.model.saver,
            )
        return scaffold
    
    # Overwrite this method if you would like to feed additional data during the training
    def train(self, *args, learning_rate=0.05, **kwargs):
        tf.logging.info("learning rate: %s" % learning_rate)
        super().train(feed_dict={"learning_rate:0": learning_rate})
    
    # Now define all parameters requested by this model 
    #   (defined in model.<some_model>.AbstractEstimator)
    @property
    def param_1(self):
        return self.get("param_1") # equal to self.run(self.model.param_1)
    @property
    def param_2(self):
        return self.get("param_2") # equal to self.run(self.model.param_2)
    
```

Some additional notes:
- estimator.get("param_1") == estimator.session.run(estimator.model.param_1)
- estimator.to_xarray(param_list) needs the PARAMS definition to export the estimated parameters as  
    xarray.Dataset()
- All necessary parameters should be directly exposed as parameter tensors in EstimatorGraph 
    (e.g. EstimatorGraph().param_1) with correct shapes as defined in PARAMS. 
    However, this property is currently not validated automatically.

