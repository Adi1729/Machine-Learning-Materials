 - ## Materials
 
    - [Machine Learning](#(Machine Learning))
    - [Python](#Python)
    - [Github Link](#Github)
    - [Data Science with R](https://supervised-ml-course.netlify.com/)

 ## Machine Learning

This repository includes materials on different topics of machine learning algorithms.
- [ ] [Hashing in ML](https://medium.com/value-stream-design/introducing-one-of-the-best-hacks-in-machine-learning-the-hashing-trick-bf6a9c8af18f)
- [ ] [Different Types of Encoders in Python](https://www.kaggle.com/discdiver/category-encoders-examples)
- [ ] [Smarter ways to encode](https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159)
- [ ] [Metrics for Linear Regression](https://towardsdatascience.com/metrics-to-understand-regression-models-in-plain-english-part-1-c902b2f4156f)
- [ ] [Adaptive Boosting](http://www.cs.princeton.edu/courses/archive/spr07/cos424/papers/boosting-survey.pdf)
- [ ] [XGBoost](http://proceedings.mlr.press/v42/chen14.pdf)
- [ ] [Random Forest](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) 
- [ ] [NLP with Deep Learning](http://web.stanford.edu/class/cs224n/syllabus.html)
- [ ] [Word Embedding - Analytics Vidhya](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)
- [ ] [Word Embedding - Xin Rong](https://www.youtube.com/watch?v=D-ekE-Wlcds)
- [ ] [CNN](http://brohrer.github.io/how_convolutional_neural_networks_work.html)
- [ ] [Weight of Evidence and Information Value](https://medium.com/@sundarstyles89/weight-of-evidence-and-information-value-using-python-6f05072e83eb)
- [ ] [Kmeans for Beginners](https://www.youtube.com/watch?v=YWgcKSa_2ag)
- [ ] [AdaBoost - Special Case of Gradient Boost](https://www.youtube.com/watch?v=ErDgauqnTHk)
- [ ] [Gradient Boost](https://www.youtube.com/watch?v=sRktKszFmSk)
- [ ] [Gradient Boost and XGBoost - Understanding Maths](https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/)
- [ ] [GBM Vs XGB](http://theprofessionalspoint.blogspot.com/2019/02/difference-between-gbm-gradient.html)
- [ ] [Decision Trees Explaination](https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249)
- [ ] [ACF and PACF : Good Explanation](https://www.youtube.com/watch?v=DeORzP0go5I&list=PLvcbYUQ5t0UHOLnBzl46_Q6QKtFgfMGc3)
- [ ] [Clearning air around Boosting](https://towardsdatascience.com/clearing-air-around-boosting-28452bb63f9e)
- [ ] [XGboost explained by StatQuest](https://www.youtube.com/watch?v=OtD8wVaFm6E&t=6s)

 ## Python
- [ ] [Pydata](https://www.youtube.com/user/PyDataTV)
- [ ] [Enthought](https://www.youtube.com/user/EnthoughtMedia) 

 ## Github 

- [ ] [Analytics Vidhya Github References](https://www.analyticsvidhya.com/blog/2018/08/best-machine-learning-github-repositories-reddit-threads-july-2018/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29)

### Machine Learning Questions

This section contains notes/summaries/questions on some of Machine Learning topics.

- [Machine Learning Interview Question Collections](https://www.kaggle.com/general/38420)
- [Data Science Interview Preparation](https://github.com/conordewey3/DS-Career-Resources/blob/master/Interview-Resources.md)
- [KMeans](#Kmeans)
- [Decorrelating Trees](#Decorrelating-Trees)
- [Shallow and Bushy Trees](#Shallow-and-Bushy-trees)
- [Random Forest](#Does-Random-Forest-overfit)
- [AdaBoost](#AdaBoost)
- [Gradient Boost](#Gradient-Boost)
- [XGBoost](#Extreme-Gradient-Boost )
- [Logistic Regression](#Logistic-Regression)

## Kmeans
  Inertia, or the within-cluster sum of squares criterion, can be recognized as a measure of how internally coherent clusters are. It suffers from various drawbacks:

   1. Inertia makes the assumption that clusters are convex and isotropic, which is not always the case. It responds poorly to elongated clusters, or manifolds with irregular shapes.                                                                            
   2. Inertia is not a normalized metric: we just know that lower values are better and zero is optimal. But in very high-dimensional spaces, Euclidean distances tend to become inflated (this is an instance of the so-called “curse of dimensionality”). Running a dimensionality reduction algorithm such as PCA prior to k-means clustering can alleviate this problem and speed up the computations.
   
   <img src="https://github.com/Adi1729/Machine-Learning-Materials/blob/master/kmeans.png">
 
## Decorrelating Trees

In Ensemble Technique (RF, GBM, GRB) , trees are decorrelated to reduce variance. Random Forest uses bagging in which number of features are selected at random and then from those features splitting criteria is decided. In this way , every tree is pretty much different from each other. 

In Boosting technique , same is done by giving weights to misclassified rows. 

## Shallow and Bushy trees

In Boosting trees, depending on problem statement one might get shallow tress as compared to those in RF. Boosting trees grow shallow trees because it can wait for later trees to grow in depth where it has not done well in terms of predictions. In Random Forest trees are independent and identically distributed , so each trees have to grow at much larger depth to identify patterns. This causes high variance which is reduced by averaging out. 

Source : https://www.youtube.com/watch?v=wPqtzj5VZus @42:05 

## Does Random Forest overfit

  No. It uses bagging technique which generates several decision trees in parallel also known as base learners. Data sampled with replacement is fed to these learners for training. The final prediction is the averaged output from all the learners. Individual tress might have high variance, but the bagging method eventually reduces variance.
  

## Sampling at every node or trees ?

Source :  https://www.researchgate.net/post/Why_values_for_SAMPLES_and_VALUE_are_different_at_each_node_of_the_tree_in_Random_Forest_scikit_python

## Adaboost

  Adaboost works by giving higher weightage to misclassification and lower weightage to correct classification.

For eg.\
Lets say total number of rows = 1000\
initial weightage to each rows = 1/1000\
correct classification =  200\
misclassification = 800\
learning rate = 0.1\
weightage to rows of correct classification = (e^-0.1)/sum of numerator = 720\
weightage to rows of incorrect classifcation = (e^0.1)/sum of numerator = 220\
Hence , 20 rows from 200 misclassfied is duplicated to get 220 rows. This might results in overfitting.\
The next model is built.\
At the end of n trees, weighted sum of predictions is taken into account.\
More is the error rate, less is the weightage given to trees.

## Gradient Boost
  
  This algorithm boost weak classifer in different ways. It uses gradient descent of loss function to reduce its misclassification. 
  
  An initial prediction is made. Its residual(Actual - Predcition) is calculated. Residual is nothing but a gradient of loss function. For the next model, this residual will be target variable. The way it differs from another algorithm like logistic is , GBM uses gradient descent for every rows rather than gradient descent at the end of each iterations. This makes the algorithm prone to outliers.
  
  Gradient Boost works well if there is good differences between classes. However, if data is noisy it might look to fit each pattern and might overfit.

## Extreme Gradient Boost      
         
   Objective Function or Gain =  F(l,Ω,λ,g,h) - Gamma
   
   F(l,Ω,λ) : Function to calculate the weight(predictions score) at every terminal nodes. This depends on loss function and Ω,λ.\
   l:  A differentiable convex loss function that measures the difference between the prediction y and the target yi.\
        One can define its own function given it's second order derivative is defined.\
    g,h : first order and second order gradient descent of a loss funciton.\
Ω : Penalizes the complexity of the model(i.e., the regression tree functions).\
λ : The additional regularization term helps to smooth the final learnt weights to avoid over-fitting.\
Gamma : This controls the number of leaves in trees.
    
  Intuitively, the regularized objective will tend to select a model employing simple and predictive functions.\
  When the regularization parameter is set to zero, the objective falls back to the traditional gradient tree boosting.
   
  Method :\
  Sort the data\
  Find the best candidate for split according to objective function (or gain) (and not gini index or entropy).\
  XGB and GBM works on greedy algorithm to decide the best split.\
  Grow the treee to maximum depth and then prunes the leaves which has negative gain.
  
  Treating Missing Values :

  Data with all missing points is guided to\
    - left nodes and gain is cacluated.\
    - right nodes and gain is calculated.
  
  And the one with maximum gain is eventually selected.
  
  Source : [Kaggle Winning Solution Xgboost Algorithm - Learn from Its Author, Tong He](https://www.youtube.com/watch?v=ufHo8vbk6g4)\
  [Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)
 
## How to see relationship of features with target variable.

  <img src="https://github.com/Adi1729/Machine-Learning-Materials/blob/master/model_interpretation_ensemble.png" width = 80%,  height = 80%>




## Logistic Regression

### Is logistic regression a linear or a non-linear model ?
Logistic regression is considered a linear model as the decision boundary would be a linear function of x i.e. the predictions can be written as linear combination of x.

if p=1/(1+ e^(-z)) and if p=0.5 is the threshold then z=0 is the decision boundary.
[link1](https://stats.stackexchange.com/questions/93569/why-is-logistic-regression-a-linear-classifier)
[link2](https://www.quora.com/Why-is-logistic-regression-considered-a-linear-model)


### Why can't we use the cost function of Linear Regression in Logistic Regression?
If we try to use the cost function of the linear regression in ‘Logistic Regression’ then it would be of no use as it would end up being a non-convex function with many local minimums, in which it would be very difficult to minimize the cost value and find the global minimum. So we define the log cost function for logistic regression which is quite convex in nature.
Below is short explaination for it.
"In case y=1, the output (i.e. the cost to pay) approaches to 0 as y_pred approaches to 1. Conversely, the cost to pay grows to infinity as y_pred approaches to 0. This is a desirable property: we want a bigger penalty as the algorithm predicts something far away from the actual value. If the label is y=1 but the algorithm predicts y_pred=0, the outcome is completely wrong."
[link1](https://www.internalpointers.com/post/cost-function-logistic-regression)
[link2](https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148)

### Can Logistic Regression be used for multiclass classification ?
Yes, using one-vs-all classification. Suppose there are 3 different classes we want to predict. We would train 3 different classifiers for each class i to predict the probability that y=i and then finally take the class that has the max probabilty while prediction.


### Is standardization required in logistic regression?
Standardization isn't required for logistic regression. The main goal of standardizing features is to help convergence of the technique used for optimization. It's just that standardizing the features makes the convergence faster.


### AIC ?
[link](https://www.methodology.psu.edu/resources/aic-vs-bic/)

### What is L1(Ridge), L2(LASSO) and Elastic Net regularization ?
Regularization is a technique to discourage the complexity of the model. It does this by penalizing the loss function. This helps to solve the overfitting problem.
In L1 regularization we change the loss function to this:

L1 regularization does feature selection. It does this by assigning insignificant input features with zero weight and useful features with a non zero weight.

L2 regularization forces the weights to be small but does not make them zero and does non sparse solution. L2 is not robust to outliers as square terms blows up the error differences of the outliers and the regularization term tries to fix it by penalizing the weights.

[link](https://medium.com/datadriveninvestor/l1-l2-regularization-7f1b4fe948f2)\
[Difference](https://discuss.analyticsvidhya.com/t/difference-between-ridge-regression-and-lasso-and-its-effect/3000)\
[ElasticNet](http://enhancedatascience.com/2017/07/04/machine-learning-explained-regularization/)


 Top 10 Data Science Blogs📈

1.	Analytics Vidya
2.	Data Science Central
3.	KDnuggets
4.	R-Bloggers
5.	Revolution Analytics
6.	Data Camp
7.	Codementor
8.	Data Plus Science
9.	Data Science 101
10.	DataRobot

🧮Learn Statistics and Probability for free📚

1.	Khan Academy
2.	OpenIntro
3.	Exam Solutions
4.	Seeing Theory
5.	Towardsdatascience
6.	Elitedatascience
7.	OLI
8.	Class Central
9.	Alison
10.	Guru99

🔏Sites with Free Data Sets🖇

1.	Data.world
2.	Kaggle
3.	FiveThirthyEight
4.	BuzzFeed
5.	Socrata OpenData
6.	Data gov
7.	Quandl
8.	Reddit or r/datasets
9.	UCI Repository
10.	Academic Torrents

📇Sites to Learn Python📕

1.	Code Academy
2.	TutorialsPoint
3.	Python org
4.	Python for Beginners
5.	Pythonspot
6.	Interactive Python
7.	Python Tutor
8.	Full Stack Python
9.	Awesome-Python
10.	CheckiO

📊Sites for Visualization📉

1.	Storytelling with Data
2.	Information is Beautiful
3.	Flowing Data
4.	Visualising Data
5.	Junk Charts
6.	The Pudding
7.	The Atlas
8.	Graphic Detail
9.	US Census and FEMA
10.	Tableau Blog

📍Best Data Science Courses Offered Online🔖

CourseEra
1.	IBM
2.	University of Michigan
3.	DeepLearning.ai
4.	Stanford Univerisity

EdX
5.	Harvard Univeristy
6.	MIT
7.	UC SanDiego
  
  
  
  
