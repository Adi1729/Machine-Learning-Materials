 - ## Materials
 
    - [Machine Learning](#(Machine Learning))
    - [Python](#Python)
    - [Github Link](#Github)

 ## Machine Learning

This repository includes materials on different topics of machine learning algorithms.

- [ ] [Adaptive Boosting](http://www.cs.princeton.edu/courses/archive/spr07/cos424/papers/boosting-survey.pdf)
- [ ] [XGBoost](http://proceedings.mlr.press/v42/chen14.pdf)
- [ ] [Random Forest](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) 
- [ ] [NLP with Deep Learning](http://web.stanford.edu/class/cs224n/syllabus.html)
- [ ] [Word Embedding - Analytics Vidhya](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)
- [ ] [Word Embedding - Xin Rong](https://www.youtube.com/watch?v=D-ekE-Wlcds)
- [ ] [CNN](http://brohrer.github.io/how_convolutional_neural_networks_work.html)
- [ ] [Information Value](https://medium.com/@sundarstyles89/weight-of-evidence-and-information-value-using-python-6f05072e83eb)
- [ ] [Kmeans for Beginners](https://www.youtube.com/watch?v=YWgcKSa_2ag)
- [ ] [AdaBoost - Special Case of Gradient Boost](https://www.youtube.com/watch?v=ErDgauqnTHk)
- [ ] [Gradient Boost](https://www.youtube.com/watch?v=sRktKszFmSk)
- [ ] [Gradient Boost and XGBoost](https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/)


 ## Python
- [ ] [Pydata](https://www.youtube.com/user/PyDataTV)
- [ ] [Enthought](https://www.youtube.com/user/EnthoughtMedia) 

 ## Github 

- [ ] [Analytics Vidhya Github References](https://www.analyticsvidhya.com/blog/2018/08/best-machine-learning-github-repositories-reddit-threads-july-2018/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29)

## Machine Learning Questions

This section contains notes/summaries/questions on some of Machine Learning topics.

### Kmeans
  Inertia, or the within-cluster sum of squares criterion, can be recognized as a measure of how internally coherent clusters are. It suffers from various drawbacks:

   1. Inertia makes the assumption that clusters are convex and isotropic, which is not always the case. It responds poorly to elongated clusters, or manifolds with irregular shapes.                                                                            
   2. Inertia is not a normalized metric: we just know that lower values are better and zero is optimal. But in very high-dimensional spaces, Euclidean distances tend to become inflated (this is an instance of the so-called “curse of dimensionality”). Running a dimensionality reduction algorithm such as PCA prior to k-means clustering can alleviate this problem and speed up the computations.
   
   <img src="https://github.com/Adi1729/Machine-Learning-Materials/blob/master/kmeans.png">
 
### Decorrelating Trees

In Ensemble Technique (RF, GBM, GRB) , trees are decorrelated to reduce variance. Random Forest uses bagging in which number of features are selected at random and then from those features splitting criteria is decided. In this way , every tree is pretty much different from each other. 

In Boosting technique , same is done by giving weights to misclassified rows. 

### Shallow and Bushy trees

In Boosting trees, depending on problem statement one might get shallow tress as compared to those in RF. This can cause high bias.  

### Does Random Forest overfit ?

  No. It uses bagging technique which generates several decision trees in parallel also known as base learners. Data sampled with replacement is fed to these learners for training. The final prediction is the averaged output from all the learners. Individual tress might have high variance, but the bagging method eventually reduces variance.
  

### Adaboost

  Adaboost works by giving higher weightage to misclassification and lower weightage to correct classification. 

For eg. 

Lets say total number of rows = 1000
initial weightage to each rows = 1/1000
correct classification =  200
misclassification = 800
learning rate = 0.1
weightage to rows of correct classification = (e^-0.1)/sum of numerator = 720
weightage to rows of incorrect classifcation = (e^0.1)/sum of numerator = 220
Hence , 20 rows from 200 misclassfied is duplicated to get 220 rows. This might results in overfitting.
The next model is built. 
At the end of n trees, weighted sum of predictions is taken into account. 
More is the error rate, less is the weightage given to trees.

### Gradient Boost
  
  This algorithm boost weak classifer in different ways. It uses gradient descent of loss function to reduce its misclassification. 
  
  An initial prediction is made. Its residual(Actual - Predcition) is calculated. Residual is nothing but a gradient of loss function. For the next model, this residual will be target variable. The way it differs from another algorithm like logistic is , GBM uses gradient descent for every rows rather than gradient descent at the end of each iterations. This makes the algorithm prone to outliers.

### Extreme Gradient Boost 

 Regularization , Penalise model for its complexity eg for number of trees or leaves by giving weights 
 Works well on sparse data (eg tfidf)
 
 Also GBM and XGB works on greedy search to decide  splitting criteria.

 
  
  
  
  
