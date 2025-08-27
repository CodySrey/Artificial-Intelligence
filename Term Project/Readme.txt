This is a python code that ran off of Visual Code Studio.
You will need to have the same imports as the code. 
To install open the Command Prompt and copy and paste the following prerequisites.

Prerequisites:
Numpy:
pip install numpy

Pandas:
pip install pandas

Matplotlib:
pip install matplotlib

Tensorflow:
pip install tensorflow

Scikit-learn:
pip install scikit-learn

If you want to check if all imports are installed copy and paste the following:
python -c "import numpy; print('NumPy Installed')"
python -c "import pandas; print('Pandas Installed')"
python -c "import matplotlib; print('Matplotlib Installed')"
python -c "import tensorflow as tf; print('TensorFlow Installed: ', tf.__version__)"
python -c "import sklearn; print('Scikit-learn Installed: ', sklearn.__version__)"

If done correctly,it should something similar to this:
" C:\Users\Mucka>python -c "import numpy; print('NumPy Installed')"
NumPy Installed

C:\Users\Mucka>python -c "import pandas; print('Pandas Installed')"
Pandas Installed

C:\Users\Mucka>python -c "import matplotlib; print('Matplotlib Installed')"
Matplotlib Installed

C:\Users\Mucka>python -c "import tensorflow as tf; print('TensorFlow Installed: ', tf.__version__)"
2024-11-20 16:51:44.302314: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-11-20 16:51:49.381607: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
TensorFlow Installed:  2.17.0

C:\Users\Mucka>python -c "import sklearn; print('Scikit-learn Installed: ', sklearn.__version__)"

In line 124-125, this is the file location for the Traffic dataset.
Be sure to change the file location in the " " to where you downloaded the dataset.
For example the file location is "C:\Users\Mucka\OneDrive\Documents\Class\AI\Traffic\Final\Trafficata.csv"
When you paste the file location in the location, change the '\' to '/'
Example: "C:/Users/Mucka/OneDrive/Documents/Class/AI/Traffic/Final/Trafficdata.csv"

Once all of the following steps are finished, you can run the code.