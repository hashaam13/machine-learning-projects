import pdb
import numpy as np
import code_for_hw3_part2 as hw3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]
features1 = [('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]

# Construct the standard data and label arrays
#auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features1)
#s=hw3.xval_learning_alg(hw3.perceptron,auto_data, auto_labels,10)
#s1=hw3.xval_learning_alg(hw3.averaged_perceptron,auto_data, auto_labels,10)
#print(s,s1)
#aa=hw3.averaged_perceptron(auto_data, auto_labels)
#print(aa[0])
#print('auto data and labels shape', auto_data.shape, auto_labels.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

# Your code here to process the auto data

#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
#review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
#review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
#dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
#review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
#review_labels = hw3.rv(review_label_list)
#print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# Your code here to process the review data
#s=hw3.xval_learning_alg(hw3.perceptron,review_bow_data, review_labels,10)
#s1=hw3.xval_learning_alg(hw3.averaged_perceptron,review_bow_data, review_labels,10)
#aa,ss=hw3.averaged_perceptron(review_bow_data, review_labels)
##tops=np.argpartition(aa[0:,0],10)
#tops=tops[0:10]
#print(tops)
#aaaa=hw3.reverse_dict(dictionary)
#for top in tops:
    #print(aaaa[top])
#print(aaaa)
#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)
# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]



# data goes into the feature computation functions

# labels can directly go into the perceptron algorithm


def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    print(x.shape)
    x=np.reshape(x,(x.shape[1]*x.shape[2],x.shape[0]))
    print(x.shape)
    return x

def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    sumed=np.mean(x,axis=2,keepdims=False)
    print(sumed)
    return (sumed.T)
    pass


def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    sumed=np.mean(x,axis=1,keepdims=False)
    return (sumed.T)
    pass


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    sumed=np.mean(x,axis=2,keepdims=False).T
    shape=sumed.shape
    print(shape)
    shape=shape[0]
    sum1=sumed[0:int(shape/2),0:]
    sum2=sumed[int(shape/2):,0:]
    sum1=np.mean(sum1,axis=0,keepdims=True)
    sum2=np.mean(sum2,axis=0,keepdims=True)
    return np.concatenate((sum1,sum2),axis=0)
    pass

# use this function to evaluate accuracy


#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------
d=[]
for i in range(10):
    d.append(mnist_data_all[i]["images"])
y0 = np.repeat(-1, len(d[9])).reshape(1,-1)
y1 = np.repeat(1, len(d[0])).reshape(1,-1)
data = np.vstack((d[9], d[0]))
labels = np.vstack((y0.T, y1.T)).T
feat=raw_mnist_features(data)
#feat1=col_average_features(data)
#feat2=top_bottom_features(data)
ac = hw3.get_classification_accuracy(feat, labels)
#ac1=hw3.get_classification_accuracy(feat1, labels)
#ac2=hw3.get_classification_accuracy(feat2, labels)
print(ac)

# Your code here to process the MNIST data

