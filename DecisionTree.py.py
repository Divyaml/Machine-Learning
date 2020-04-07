"""

Divya Machenahalli Lokesh, DXM190018
Nidhin Anisham, NXA190000
    
"""

# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.


import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix
import graphviz

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # THIS METHOD IS NOT USED
    part_dict = {}
    for i,j in enumerate(x):

        print(j)
        if j in part_dict:
            part_dict[j].append(i)
        else:
            part_dict[j] = [i]
    return part_dict


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    
    #get unique vaues of y (i.e 0,1) and their counts
    unique_y, count_y = np.unique(y, return_counts=True)
    total = 0
    
    #loop for finding total entropy of positive and negative examples
    for i in range(len(unique_y)):
        cal = -(count_y[i]/len(y))*math.log2(count_y[i]/len(y))
        total += cal
    return total


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    
    y_entropy = entropy(y)
    y1 = []
    y2 = []
    
    #loop for splitting examples of y where x=1
    for i,j in enumerate(x):    
        if j==1:
            y1.append(y[i])
        else:
            y2.append(y[i])
    y1 = np.array(y1)
    y2 = np.array(y2)
    
    #calculating conditional entropy of y given x
    y_x_entropy = ((len(y1)/len(y))*entropy(y1))+((len(y2)/len(y))*entropy(y2))
    
    #returning the mutual information
    return y_entropy-y_x_entropy


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    
    unique_y, count_y = np.unique(y, return_counts=True)
    
    #termination conditions  
    if(unique_y.size == 1):   #check if all y values are the same
        return unique_y[0]
    if(attribute_value_pairs == None):  #check if there are no atribute value pairs left
        return unique_y[np.argmax(count_y)]
    if(depth == max_depth):  #check if maximum allowed depth has been reached
        return unique_y[np.argmax(count_y)]
    
    max_info_gain = -1  #assigning -1 as information gain >= 0
    
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    
    #loop to calculate information gain for each attribute value pairs
    for i,j in enumerate(attribute_value_pairs):
        new_x = x[:,j[0]].copy()
        feature = abstract_column(new_x,j[1])
        info_gain = mutual_information(feature,y)
        if(info_gain>max_info_gain):  #gets the attribute value pair with maximum information gain
            max_info_gain = info_gain
            max_attribute = i
            max_attribute_tuple = (attribute_value_pairs[i][0],attribute_value_pairs[i][1])
    
    #loop to split the data based on the attribute chosen. 
    #(x1,y1) split for attribute = True and the (x2,y2) for attribute = False        
    for i,j in enumerate(x):
        if(j[attribute_value_pairs[max_attribute][0]]==attribute_value_pairs[max_attribute][1]):
            x1.append(j)
            y1.append(y[i])
        else:
            x2.append(j)
            y2.append(y[i])
            
    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)
    
    #key tuple for attribute = True
    pair_tuple_t = (attribute_value_pairs[max_attribute][0],attribute_value_pairs[max_attribute][1], True)    
    
    #key tuple for attribute = False
    pair_tuple_f = (attribute_value_pairs[max_attribute][0],attribute_value_pairs[max_attribute][1], False)
    
    #get new attribute value pairs for x1 split
    attribute_value_pairs_x1 = get_attribute_value_pairs(x1)
    for i,j in enumerate(attribute_value_pairs_x1):
        if j[0] == max_attribute_tuple[0] and j[1] == max_attribute_tuple[1]:
            max_attribute = i
            break
    attribute_value_pairs_x1.pop(max_attribute) #pop attribute that has already been processed
    
    #get new attribute value pairs for x2 split
    attribute_value_pairs_x2 = get_attribute_value_pairs(x2)
    
    #recursive call on attribute = True and attribute = False
    return {pair_tuple_t: id3(x1,y1,attribute_value_pairs_x1,depth+1,max_depth),
            pair_tuple_f: id3(x2,y2,attribute_value_pairs_x2,depth+1,max_depth)}

def abstract_column(x,val):
    """ 
    Abstracts column x. If x is equal to the value given, x=1 else x=0.
    Returns abstracted column
    """
    
    for i in range(len(x)):
        if(x[i] == val):
            x[i] = 1
        else:
            x[i] = 0
    return x

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    
    if tree==1 or tree==0: #terminate conditions if leaf node is reached
        return tree
    else:
        for i,j in tree.items():
            if((i[1]==x[i[0]])==i[2]):
                return predict_example(x,tree[i])
    
def compute_error(y_true, y_pred): 
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    error_sum=0
    for i in range(len(y_true)):
        if y_true[i]!=y_pred[i]:
            error_sum+=1
    error= (1/len(y_true))*error_sum
    return error
    
def confusion_matrix(y_true,y_pred):
    """
    Computes the confusion matrix.
    Returns 2d list with the format [[True Positives,False Negatives],[False Positives,True Negatives]]
    """
    
    true_p=0
    true_n=0
    false_p=0
    false_n=0
    for i in range(len(y_true)):
        if y_true[i]==1 and y_pred[i]==1:  #true positive
            true_p+=1
        elif y_true[i]==1 and y_pred[i]==0:  #false negative
            false_n+=1
        elif y_true[i]==0 and y_pred[i]==1:  #false positive
            false_p+=1
        elif y_true[i]==0 and y_pred[i]==0:  #true negative
            true_n+=1
    confusion_matrix=np.array([[true_p,false_n],[false_p,true_n]])
    return confusion_matrix
            

def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

def get_attribute_value_pairs(Xtrn):
    """
    Computes all the possibile attribute value pairs in the matrix
    Returns list of tuples with (attribute,value)
    """
    
    avp = []
    col = np.size(Xtrn,1)
    for i in range(col):
        x = np.unique(Xtrn[:,i])  #gets unique values in a column
        for j in x:
            avp.append((i,j))
    return avp

def discretize(x):
    """
    This function performs discretization of continuous data
    If the value of the data < mean of the corresponding column, data = 0 else data = 1
    """
    
    mean = np.mean(x,axis=0)
    for i in range(np.size(x,1)):
        x_list = x[:,i]
        for j in range(len(x_list)):
            if x_list[j] < mean[i]:
                x_list[j] = 0
            else:
                x_list[j] = 1

if __name__ == '__main__':
    
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn1 = M[:, 0]
    Xtrn1 = M[:, 1:]

    M = np.genfromtxt('./monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn2 = M[:, 0]
    Xtrn2 = M[:, 1:]
    
    M = np.genfromtxt('./monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn3 = M[:, 0]
    Xtrn3 = M[:, 1:]
    
    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst1 = M[:, 0]
    Xtst1 = M[:, 1:]
    
    M = np.genfromtxt('./monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst2 = M[:, 0]
    Xtst2 = M[:, 1:]
    
    M = np.genfromtxt('./monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst3 = M[:, 0]
    Xtst3 = M[:, 1:]
    
    monk1tst_error=[]
    monk1trn_error=[]
    monk2tst_error=[]
    monk2trn_error=[]
    monk3tst_error=[]
    monk3trn_error=[]
    
    total_depth = 10  #total depth of tree to consider
    
    #loop to run id3 algorithm on depths 1 to 10 and get error for each depth
    for i in range(total_depth):
        attribute_value_pairs = get_attribute_value_pairs(Xtrn1)
        decision_tree = id3(Xtrn1, ytrn1, attribute_value_pairs, max_depth=i+1)
        y_pred = [predict_example(x, decision_tree) for x in Xtst1]
        tst_err = compute_error(ytst1, y_pred)
        monk1tst_error.append(tst_err*100)
        
        if(i==1):
            tree1 = decision_tree
            con_mat1=confusion_matrix(ytst1,y_pred)
        elif(i==2):
            tree2 = decision_tree
            con_mat2=confusion_matrix(ytst1,y_pred)
            
        y_pred = [predict_example(x, decision_tree) for x in Xtrn1]
        tst_err = compute_error(ytrn1, y_pred)
        monk1trn_error.append(tst_err*100)
        
        
        attribute_value_pairs = get_attribute_value_pairs(Xtrn2)
        decision_tree = id3(Xtrn2, ytrn2, attribute_value_pairs, max_depth=i+1)
        y_pred = [predict_example(x, decision_tree) for x in Xtst2]
        tst_err = compute_error(ytst2, y_pred)
        monk2tst_error.append(tst_err*100)
        y_pred = [predict_example(x, decision_tree) for x in Xtrn2]
        tst_err = compute_error(ytrn2, y_pred)
        monk2trn_error.append(tst_err*100)
        
        attribute_value_pairs = get_attribute_value_pairs(Xtrn3)
        decision_tree = id3(Xtrn3, ytrn3, attribute_value_pairs, max_depth=i+1)
        y_pred = [predict_example(x, decision_tree) for x in Xtst3]
        tst_err = compute_error(ytst3, y_pred)
        monk3tst_error.append(tst_err*100)
        y_pred = [predict_example(x, decision_tree) for x in Xtrn3]
        tst_err = compute_error(ytrn3, y_pred)
        monk3trn_error.append(tst_err*100)
    
    
    #printing average error for 10 depths
    print("Average training error for monks-1: {:.2f}%".format(sum(monk1trn_error)/len(monk1trn_error))) 
    print("Average test error for monks-1: {:.2f}%".format(sum(monk1tst_error)/len(monk1tst_error)))  
    print("\nAverage traning error for monks-2: {:.2f}%".format(sum(monk2trn_error)/len(monk2trn_error)))  
    print("Average test error for monks-2: {:.2f}%".format(sum(monk2tst_error)/len(monk2tst_error))) 
    print("\nAverage training error for monks-3: {:.2f}%".format(sum(monk3trn_error)/len(monk3trn_error)))  
    print("Average test error for monks-3: {:.2f}%".format(sum(monk3tst_error)/len(monk3tst_error)))  
    
    #using matplotlib to plot error curves
    x=np.arange(1,total_depth+1,1)
    fig, axes = plt.subplots(nrows=1,ncols=3)
    axes[0].plot(x,monk1tst_error,color='blue',label='Test_data')
    axes[0].plot(x,monk1trn_error,color='red',label='Train_data')
    axes[1].plot(x,monk2tst_error,color='orange',label='Test_data')
    axes[1].plot(x,monk2trn_error,color='purple',label='Train_data')
    axes[2].plot(x,monk3tst_error,color='green',label='Test_data')
    axes[2].plot(x,monk3trn_error,color='blue',label='Train_data')
    fig.suptitle('Error Curves for Test Vs Traning Data',fontsize=10,color='Blue')
    for i in range(3):
        axes[i].legend()
        axes[i].set_xticks(x)
        axes[i].set_yticks(np.arange(0,60,10))
        axes[i].set_xlabel('Depth from 1-10' )
        axes[i].set_ylabel('Error %')
        axes[i].set_title('Monk'+str(i+1))        
    
    print("\n--------------------------------------------------------------\n")    
    #part-b    
    print("\nmonks-1 Confusion Matrix for depth 1:")
    print(con_mat1)
    
    print("\nmonks-1 Tree for depth 1:")
    if type(tree1)==dict:
            visualize(tree1)
    else:
        print("monks-1 Tree is a single value "+tree1)
        
    print("\nmonks-1  Confusion Matrix for depth 2:")
    print(con_mat2)
    
    print("\nmonks-1  Tree for depth 2:")
    if type(tree2)==dict:
            visualize(tree2)
    else:
        print("monks-1 Tree is a single value "+tree2)
    
    print("\n--------------------------------------------------------------\n")    
    #part-c
    model = tree.DecisionTreeClassifier() #initialize the model
    model = model.fit(Xtrn1,ytrn1) #fit the training data
    
    y = model.predict(Xtst1) #get predictions of y on the test data
    
    #calculating the error by comparing predicted value with actual value
    count = 0
    for i in range(len(y)):
        if(y[i]==ytst1[i]):
            count += 1
    print("\nTraining the model using sklearn:")
    print("Accuracy: {:.2f}%".format((count/len(y))*100))
    
    print("\nmonks-1 Confusion Matrix using sklearn:")
    print(confusion_matrix(ytst1,y))  #print confusion matrix using sklearn metrics
    
    #using graphviz to render the tree onto a pdf
    dot_data = tree.export_graphviz(model,out_file=None)
    graph = graphviz.Source(dot_data) 
    graph.render('monks-1_dtree',view=True)
    
    print("\n--------------------------------------------------------------\n")
    #part-d
    print("\nTraining the model using Banknote Authentication dataset.")
    # Load the training data
    M = np.genfromtxt('./data_banknote_authentication.txt', missing_values=0, skip_header=0, delimiter=',', dtype=float)
    np.random.shuffle(M)  #shuffle the data
    train_size = round(0.8*len(M)) #set size of training data
    Xtrn, Xtst = M[:train_size,:-1], M[train_size:,:-1]
    ytrn,ytst = M[:train_size,-1], M[train_size:,-1]
    
    #discretize the continuous values for the features
    discretize(Xtrn)
    discretize(Xtst)
    
    attribute_value_pairs = get_attribute_value_pairs(Xtrn)
    
    #run the id3 decision tree algorithm for train with depth = 1
    decision_tree = id3(Xtrn, ytrn, attribute_value_pairs, max_depth=1)
    
    #get predicitions for test set
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    
    #prints confusion matrix
    con_mat=confusion_matrix(ytst,y_pred)
    print("\nConfusion Matrix for depth 1:")
    print(con_mat)
    
    print("\nTree for depth 1:")
    if type(decision_tree)==dict:
            visualize(decision_tree)
    else:
        print("Tree could not be visualized")
        
    attribute_value_pairs = get_attribute_value_pairs(Xtrn)
    print(attribute_value_pairs)
    #run the id3 decision tree algorithm for train with depth = 2
    decision_tree = id3(Xtrn, ytrn, attribute_value_pairs, max_depth=2)
    
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    
    print()
    
    con_mat=confusion_matrix(ytst,y_pred)
    print("\nConfusion Matrix for depth 2:")
    print(con_mat)
    
    print("\nTree for depth 2:")
    if type(decision_tree)==dict:
            visualize(decision_tree)
    else:
        print("Tree could not be visualized")
    
    print("\nTraining the model using sklearn:")
    model = tree.DecisionTreeClassifier() #initialize the model
    model = model.fit(Xtrn,ytrn) #fit the training data
    
    y = model.predict(Xtst) #get predictions of y on the test data
    
    #calculating the error by comparing predicted value with actual value
    count = 0
    for i in range(len(y)):
        if(y[i]==ytst[i]):
            count += 1
    print("Accuracy: {:.2f}%".format((count/len(y))*100))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(ytst,y))  #print confusion matrix using sklearn metrics
    
    #using graphviz to render the tree onto a pdf
    dot_data = tree.export_graphviz(model,out_file=None)
    graph = graphviz.Source(dot_data) 
    graph.render('note_authentication_dtree',view=True)