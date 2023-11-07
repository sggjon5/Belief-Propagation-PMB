# Belief Propagation - PMB
Belief propagation - Poisson multi-Bernoulli (BP-PMB) Filtering on multiple targets where the marginal data association probabilities are approximated via belief propagation, translated code from J. Williams' paper https://arxiv.org/abs/1203.2995

This code has been translated using chatGPT - 3.5 from the original MATLAB scripts
provided as ancillary files to https://arxiv.org/abs/1203.2995 by Jason L. Williams.

The translated files were then edited by George Jones as the translation, whilst correct in
majority, fall foul of some common errors that prevent the code from running.

Appended to the translated code is the calucaltion of the GOSPA error, with the code
for these calculations being provided in https://github.com/ewilthil/gospapy/tree/master/gospapy

Along with this GOSPA error calculation, some code has been added to plot the GOSPA error and its 
constituent parts (localisation, missed and false targets).

To execute the code, run 'main.py'.

References:
    
    - J. Williams, Marginal multi-Bernoulli filters: RFS derivation of MHT, 
      JIPDA and association-based MeMBer, IEEE Transactions on Aerospace and 
      Electronic Systems, 2015
      
      
    - J. Williams and R. Lau, Approximate evaluation of marginal association
    probabilities with belief propagation, IEEE Transactions on Aerospace and 
    Electronic Systems , 2014
    
    
    - A. S. Rahmathullah, A. F. Garcia-Fernandez and L. Svensson, Generalized
      optimal sub-pattern assignment metric, 20th International Conference on
      Information Fusion, 2017.
    

@author: George Jones
