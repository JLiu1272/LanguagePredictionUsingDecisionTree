# Own decision tree class
from decisionTreeTrain import *

def adaBoostTrain_main(csv_file="processed_data/binary_dutch_eng.csv", num_iter=3):
    """
    csv_file - the file with the training data
    :param csv_file:
    :return:
    """
    # Parse the csv file as data
    datas = pd.read_csv(csv_file)

    # Save the target variable
    y = datas["Lang"]

    # Drop target
    datas = datas.drop(["Lang"], axis=1)

    # Note the total number of datas there are
    N, num_attr = datas.shape

    print("Num Samples")
    print(N)

    # Initialise the weights as 1/N
    weights = np.ones(N) / N

    stump_val = []

    hypot_weights = [1] * num_iter

    for t in range(num_iter):
        print()
        print("Iter: " + str(t))
        # Generate a decision stump given then the weight
        # save this tree structure
        h = decisionTreeFit(depth=1, weight=weights)

        print("Best Attribute")
        print(h.attribute)

        print("Modified Weights: ")
        print(weights)

        # Predict using the decision stump generated
        # get the result of prediction
        pred = dt_predict(h, datas)

        error = 0

        # Identify the results that did not
        # match the true results
        # Set the error of those to
        # be higher
        for i in range(len(pred)):
            if pred[i] != y.iloc[i]:
                error = error + weights[i]
            else:
                weights[i] = weights[i] * error / (1 - error)

        print("Error Rate: ")
        print(error)

        total = 0
        # Normalize the weights
        for weight in weights:
            total += weight

        for i in range(N):
            weights[i] = weights[i] / total

        # Update the hypothesis weight for that index
        hypot_weights[t] = math.log(((1 - error) / (error)),2)

        # Append the stump to the stumps list
        stump_val.append(h)

    return hypot_weights, stump_val


def ada_predict(test_csv, hypot_weights, stump_val):
    '''
    Make a prediction and calculate the
    accuracy
    :param test_csv:
    :param hypot_weights:
    :param stump_val:
    :return:
    '''
    # Parse the csv file as data
    tests = pd.read_csv(test_csv)

    # Save the target variable
    y = tests["Lang"]

    # Drop target
    tests = tests.drop(["Lang"], axis=1)

    summation = 0
    correct = 0
    incorrect = 0

    '''
    Traverse through every test. If the prediction
    was 0, negate the summation. If the prediction
    was 1, increase the weight 
    '''
    for test_idx in range(tests.shape[0]):
        for stump_idx in range(len(stump_val)):
            # Make the prediction using the saved decision stump
            pred = dt_predict(stump_val[stump_idx], tests.iloc[test_idx], single=True)
            if pred == 0:
                summation += -1 * hypot_weights[stump_idx]
            elif pred == 1:
                summation += 1 * hypot_weights[stump_idx]

        # If the final summation is greater than 0,
        # and the classifier predicts that it is
        # english, otherwise it predicts that
        # it is Dutch
        if summation > 0:
            # If prediction and label matches
            # it is correct
            if y.iloc[test_idx] == 1:
                correct += 1
            else:
                incorrect += 1
        else:
            if y.iloc[test_idx] == 0:
                correct += 1
            else:
                incorrect += 1

    return correct/y.shape[0]

def ada_complete_predict(examples="processed_data/train.txt", hypothesisOut="adaBoostClassifier.py"):
    '''
    The main function that
    does all of the ada-boosting
    training
    :return:
    '''
    hypot_weights, stump_val = adaBoostTrain_main(num_iter=8)

    # Write the result of the adaboost function the
    # adaboost classifier
    with open(hypothesisOut, 'w') as result_csv:
        writer = csv.writer(result_csv)
        writer.writerow(['hypot_weights', 'stump_val'])
        writer.writerow([hypot_weights, stump_val])
        result_csv.close()

    # Convert example file into a csv file
    write_to_csv("processed_data/binary_dutch_eng.csv", examples)

    print(ada_predict("processed_data/binary_dutch_eng_validation.csv", hypot_weights, stump_val))
    #ada_predict("processed_data/binary_dutch_eng_validation.csv", "", "")


