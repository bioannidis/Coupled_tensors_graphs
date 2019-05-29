# Coupled_tensors_graphs
This software implements the main algorithm found in the paper
Coupled Graphs and Tensor Factorization for Recommender Systems and Community Detection
Vassilis N. Ioannidis, Student member, IEEE, Ahmed S. Zamzam, Student member, IEEE, 
Georgios B. Giannakis, Fellow, IEEE, and Nicholas D. Sidiropoulos, Fellow, IEEE

Paper link: https://arxiv.org/abs/1809.08353 

CGTF_wrapper.m:
Implements the alternating direction method of multipliers found in section 3.1 of the aforementioned paper
CGTF_example.m:
Contains an example on how to process the Digg.mat, generate the training data and call the CGTF_wrapper.m 

Digg.mat:
The Digg dataset includes stories, and users along with their time-stamped actions with 
respect to stories, as well as the social network of users. In addition, a set of keywords
is assigned to each story. After discretizing the time into 20 time intervals over
3 days, we construct a tensor comprising the number of comments that user i wrote on story j during the k-th
time interval stored in the (i, j, k) item. Also, a story-story graph is constructed where any two stories are connected
only if they share more than two keywords.
