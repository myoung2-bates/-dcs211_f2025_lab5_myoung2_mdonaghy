import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns   # yay for Seaborn plots!
import matplotlib.pyplot as plt
import random

###########################################################################
def drawDigitHeatmap(pixels: np.ndarray, showNumbers: bool = True) -> None:
    ''' Draws a heat map of a given digit based on its 8x8 set of pixel values.
    Parameters:
        pixels: a 2D numpy.ndarray (8x8) of integers of the pixel values for
                the digit
        showNumbers: if True, shows the pixel value inside each square
    Returns:
        None -- just plots into a window
    '''

    (fig, axes) = plt.subplots(figsize = (4.5, 3))  # aspect ratio

    rgb = (0, 0, 0.5)  # each in (0,1), so darkest will be dark blue
    colormap = sns.light_palette(rgb, as_cmap=True)    
    # all seaborn palettes: 
    # https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f

    # plot the heatmap;  see: https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # (fmt = "d" indicates to show annotation with integer format)
    sns.heatmap(pixels, annot = showNumbers, fmt = "d", linewidths = 0.5, \
                ax = axes, cmap = colormap)
    plt.show(block = False)

###########################################################################
def fetchDigit(df: pd.core.frame.DataFrame, which_row: int) -> tuple[int, np.ndarray]:
    ''' For digits.csv data represented as a dataframe, this fetches the digit from
        the corresponding row, reshapes, and returns a tuple of the digit and a
        numpy array of its pixel values.
    Parameters:
        df: pandas data frame expected to be obtained via pd.read_csv() on digits.csv
        which_row: an integer in 0 to len(df)
    Returns:
        a tuple containing the reprsented digit and a numpy array of the pixel
        values
    '''
    digit  = int(round(df.iloc[which_row, 64]))
    pixels = df.iloc[which_row, 0:64]   # don't want the rightmost rows
    pixels = pixels.values              # converts to numpy array
    pixels = pixels.astype(int)         # convert to integers for plotting
    pixels = np.reshape(pixels, (8,8))  # makes 8x8
    return (digit, pixels)              # return a tuple

###################
def cleanTheData(dataframe): 
    '''function that takes a pandas dataframe and cleans it, returning a numpy array of the cleaned data
    
        parameters: 
            - pandas dataframe 
        
        returns: 
            - numpy array
            
    '''
    col65name = dataframe.columns[65]  # get column name at index 65 

    df_clean = dataframe.drop(columns=[col65name])  # drop by name is typical, but what else is possible?
    
    df_clean.info()

    return df_clean 

def predictiveModel(train_set, features) -> int:
    ''' function that takes a training set and features for a given digit and returns a predicted digit
    
        paramters: 
            - numpy array of training data
            - numpy array of test features
            
        returns: 
            - preditced digit        
    '''

    test_features = np.asarray(features) # make numpy array of test features 

    # using Euclidean distance to find one closest digit 
    dist = np.linalg.norm 

    num_rows, num_cols = train_set.shape  # data size

    closest_digit   = train_set[0]
    closest_features = train_set[0,0:64] # feautures include everything but last column (labels)
    closest_distance = dist(test_features - closest_features)

    for i in range(1, num_rows):
        current_features = train_set[i, 0:64] 
        current_distance = dist(test_features - current_features)

        if current_distance < closest_distance:
            closest_distance = current_distance  # remember closest!
            closest_digit   = train_set[i]

    # done with comparison, now have best prediction 
    predicted_digit = int(closest_digit[-1])  
    return predicted_digit
    
def splitData(all_data: np.ndarray) -> list: 
    ''' 
    function that accepts a numpy array full data set and creates training and testing data sets
    parameters: 
     - all_data: numpy array of full dataset

    returns: 
     - list of: X_test, y_test, X_train, y_train  
     '''
     # defining features and lables for kNN 
    X_all = all_data[:,:-1] # features (all rows, columns 0-second to last)
    y_all = all_data[:,-1] # labels (all rows, last column only)

    indices = np.random.permutation(len(y_all)) # indices is a permutation list 

    # scramble x and y with permutation 
    X_labeled = X_all[indices]
    y_labeled = y_all[indices]

    # train on 80%, test on 20% 
    num_rows = X_labeled.shape[0]
    test_percent = 0.20
    test_size = int(test_percent*num_rows)

    X_test = X_labeled[:test_size] # testing features 
    y_test = y_labeled[:test_size] # testing labels 

    X_train = X_labeled[test_size:] # training features 
    y_train = y_labeled[test_size:] # training labels 

    return [X_test, y_test, X_train, y_train] 

# compare labels function code from tutorial 
def compareLabels(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> int:
    ''' a more neatly formatted comparison, returning the number correct '''
    num_labels = len(predicted_labels)
    num_correct = 0

    for i in range(num_labels):
        predicted = int(round(predicted_labels[i]))  # round-to-int protects from float imprecision
        actual    = int(round(actual_labels[i]))
        result = "incorrect"
        if predicted == actual:  # if they match,
            result = ""       # no longer incorrect
            num_correct += 1  # and we count a match!

    print()
    print(f"Correct: {num_correct} out of {num_labels}")
    return num_correct

def findBestK (X_train, y_train, X_test, y_test) -> int:
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    '''
    A function that determines the best value of k for kNN by testing multiple random seeds
    parameters: 
    - X_train: numpy array of training features
    - y_train: numpy array of training labels

    Returns: 
    -best_k: integer of k that has highest average accuracy

    '''
    seeds = [8675309, 5551212, 6767676]
    k_values = [2, 4, 6, 8, 10, 12, 14] #try even k's from 1-15
    all_scores = []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        
        #spilt into 20% for validation and 80% for training
        X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

        for k in k_values: 
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            acc = model.score(X_validate, y_validate)
            all_scores.append((seed, k, acc))
    
    #find best avg accuracy for each k
    average_scores = {}
    for result in all_scores: 
        seed, k, accuracy = result

    # If this k value hasn’t been seen yet, start a new list for it
        if k not in average_scores:
            average_scores[k] = [accuracy]
        else:
        # If we already have this k, add another accuracy score to its list
            average_scores[k].append(accuracy)

# Now compute the average accuracy for each k value
    best_k = None
    best_avg = 0

    for k in average_scores:
        avg_accuracy = np.mean(average_scores[k])

    # Check if this k is better than what we’ve seen before
    if avg_accuracy > best_avg:
        best_avg = avg_accuracy
        best_k = k
    return best_k

def trainAndTest(X_train, y_train, X_test, y_test, best_k):
    from sklearn.neighbors import KNeighborsClassifier
    '''
    Train and test a kNN model using the best determined value of k.
    Parameters:
    - X_train (array-like): Training feature data
    - y_train (array-like): Training labels
    - X_test (array-like): Testing feature data
    - y_test (array-like): True labels for the test set
    - best_k (int): Best k value determined from prior experiments

    Returns:
        predicted_labels (ndarray): Labels predicted by the trained model
    """
    '''
    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train, y_train)

    #Test the model
    predicted_labels = knn_model.predict(X_test)

    # Compute accuracy
    accuracy = np.mean(predicted_labels == y_test)
    print(f"Model trained and tested with k = {best_k}")
    print(f"Prediction accuracy: {accuracy}")

    #Compare predicted vs actual labels
    compareLabels(y_test, predicted_labels)

    return predicted_labels

###################
def main() -> None:
    # for read_csv, use header=0 when row 0 is a header row
    filename = 'digits.csv'
    df = pd.read_csv(filename, header = 0)
    print(df.head())
    print(f"{filename} : file read into a pandas dataframe...")

    df_clean = [] 

    df_clean = cleanTheData(df)

    num_to_draw = 5
    for i in range(num_to_draw):
        # let's grab one row of the df at random, extract/shape the digit to be
        # 8x8, and then draw a heatmap of that digit
        random_row = random.randint(0, len(df) - 1)
        (digit, pixels) = fetchDigit(df, random_row)

        print(f"The digit is {digit}")
        print(f"The pixels are\n{pixels}")  
        drawDigitHeatmap(pixels)
        plt.show()

    #
    # OK!  Onward to knn for digits! (based on your iris work...)

    # convert data to numpy array 
    final_df = df_clean.to_numpy()
    # defining features and lables for kNN 
    X_all = final_df[:,0:64] # features (all rows, columns 0-63)
    Y_all = final_df[:,64] # labels (all rows, column 64 only)

    indices = np.random.permutation(len(Y_all)) # indices is a permutation list 

    # scramble x and y with permutation 
    X_labeled = X_all[indices]
    Y_labeled = Y_all[indices]
    print(X_labeled)
    print(Y_labeled)

    # train on 80%, test on 20% 
    num_rows = X_labeled.shape[0]
    test_percent = 0.20
    test_size = int(test_percent*num_rows)

    X_test = X_labeled[:test_size] # testing features 
    Y_test = Y_labeled[:test_size] # testing labels 

    X_train = X_labeled[test_size:] # training features 
    Y_train = Y_labeled[test_size:] # training labels 

    num_train_rows = len(Y_train)
    num_test_rows = len(Y_test)
    print(f"total_rows: {num_rows}; training with {num_train_rows} rows; testing with {num_test_rows} rows")

    # numpy arrays of training and testing data 
    train_set = np.column_stack((X_train, Y_train)) # 80% 
    test_set = np.column_stack((X_test, Y_test)) # 20% 
 
    # importing progress bar capability 
    from tqdm import tqdm 

    # test first predicitveModel 
    predictions = []
    correct = 0 

    for i in tqdm(range(len(X_test)), desc="Predicting"): # added progress bar
        test_features = X_test[i]               # one row of features (no label)
        predicted_label = predictiveModel(train_set, test_features)
        true_label = Y_test[i] 

        predictions.append(predicted_label)

        if predicted_label == true_label: 
            correct += 1 

    print(predictions)

    accuracy = correct / len(Y_test)
    print(f"Accuracy: {accuracy:.3f}")
    
    # now training with 20% and testing with 80% 
    num_rows = X_labeled.shape[0]
    test_percent = 0.80
    test_size = int(test_percent*num_rows)

    X_test = X_labeled[:test_size] # testing features 
    Y_test = Y_labeled[:test_size] # testing labels 

    X_train = X_labeled[test_size:] # training features 
    Y_train = Y_labeled[test_size:] # training labels 

    num_train_rows = len(Y_train)
    num_test_rows = len(Y_test)
    print(f"total_rows: {num_rows}; training with {num_train_rows} rows; testing with {num_test_rows} rows")

    # numpy arrays of training and testing data 
    train_set = np.column_stack((X_train, Y_train)) # 80% 
    test_set = np.column_stack((X_test, Y_test)) # 20% 
 
 # second test predicitveModel 
    predictions = []
    correct = 0 
    incorrect = []

    for i in tqdm(range(len(X_test)), desc="Predicting"): # added progress bar
        test_features = X_test[i]               # one row of features (no label)
        predicted_label = predictiveModel(train_set, test_features)
        true_label = Y_test[i] 

        predictions.append(predicted_label)

        if predicted_label == true_label: 
            correct += 1 

        else: 
            # store wrong predictions 
            incorrect.append((i, true_label, predicted_label))

    print(predictions)

    accuracy = correct / len(Y_test)
    print(f"Accuracy: {accuracy:.3f}")

    # heat mapping wrong predictions 
    incorrect_map = incorrect[:5]
    for test_index, actual, predicted in incorrect_map: 
        original_index = indices[test_index] # test row location in original df 
        digit, pixels = fetchDigit(df_clean, original_index)
        drawDigitHeatmap(pixels)
        plt.show()

    # calling splitData function with original numpy full dataset 
    X_test, y_test, X_train, y_train = splitData(final_df)

    # running k-NN classifier using guessed value of k 
    from sklearn.neighbors import KNeighborsClassifier

    k = 60 # guess 
    knn_model = KNeighborsClassifier(n_neighbors = k) # k is set to 60 

    # train the model with this k 
    knn_model.fit(X_train, y_train) 

    # testing this model 
    predicted_labels = knn_model.predict(X_test) 
    actual_labels = y_test

    # printing results of this test 
    compareLabels(predicted_labels, actual_labels)

    #Find the best k value
    best_k = findBestK(X_train, y_train, X_test, y_test)

    #Train and test using that best k
    predicted_labels = trainAndTest(X_train, y_train, X_test, y_test, best_k)

###############################################################################
# wrap the call to main inside this if so that _this_ file can be imported
# and used as a library, if necessary, without executing its main
if __name__ == "__main__":
    main()
