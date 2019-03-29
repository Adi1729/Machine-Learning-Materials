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


 ## Python
- [ ] [Pydata](https://www.youtube.com/user/PyDataTV)
- [ ] [Enthought](https://www.youtube.com/user/EnthoughtMedia) 

 ## Github 

- [ ] [Analytics Vidhya Github References](https://www.analyticsvidhya.com/blog/2018/08/best-machine-learning-github-repositories-reddit-threads-july-2018/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29)

## Machine Learning Questions

This section contains notes/summaries/questions on some of Machine Learning topics.

### Kmeans
  Inertia, or the within-cluster sum of squares criterion, can be recognized as a measure of how internally coherent clusters are. It suffers from various drawbacks:

    Inertia makes the assumption that clusters are convex and isotropic, which is not always the case. It responds poorly to elongated clusters, or manifolds with irregular shapes.
    Inertia is not a normalized metric: we just know that lower values are better and zero is optimal. But in very high-dimensional spaces, Euclidean distances tend to become inflated (this is an instance of the so-called “curse of dimensionality”). Running a dimensionality reduction algorithm such as PCA prior to k-means clustering can alleviate this problem and speed up the computations.

