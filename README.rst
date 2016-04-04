word_spotting
=============

C++ application to perform keyword spotting on GW/PAR/IAM databases.

* Commands:

  * train: Train the selected network (See config.hpp) and test the results
  * features: export all the features using the current configuration
  * evaluate: Simply evaluate the current netowrk (loading the weights)
  * evaluate_features: Simply evaluate features using DTW
