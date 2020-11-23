# Capstone Model Pickling

A short example showing a pickled pipeline. This is not intended to be forked when working on your Capstone project, but rather to serve as an example when working on pickling a model.

### Note on `.py` Files

This example shows the pickling being done in a `.py` file (`model_pickling.py`) rather than a Jupyter Notebook file on purpose. Especially if you are using custom classes (transformers, metrics, etc.) you will run into problems with the module imports when pickling in a notebook.

The best technique is to put any custom classes in a separate file that can be imported, then also copy that file so it is available in the final deployment context.  In this case, it's in `src/models/custom_transformers.py`.  The reason for the path is that this will be the eventual import path from `app.py` in the Flask app.

We recommend that you use a notebook to develop your code, then ultimately run it in a `.py` file when creating the final pickled model.

### Note on Pipelines

Ideally, when you are the point of deploying the model you will have the model preprocessing as part of a pipeline.  If you aren't at this stage, you will need to pickle each preprocessor (e.g. `StandardScaler`) separately, then load it along with your model as part of preprocessing code in your final deployed solution.
