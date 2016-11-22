import tensorflow as tf
if tf.__version__ == "0.11.0rc2":
    print "SUCCESS"
else:
    print "FAIL: TensorFlow not the right version."

