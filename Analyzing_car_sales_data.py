# -*- coding: utf-8 -*-
"""
Quick introduction to the 93cars data.

Here we try to predict MidrangePrice from HighwayMPG.

-Doug Galarus, CS 5665, Spring 2019
"""

import pandas
import random
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

# Read in the data using Pandas. The result is stored in a Pandas Data Frame.
df = pandas.read_csv("93cars.csv")

def question1():
    # Assign to X and y. We have to reshape X to match
    # what the subsequent method expects.
    X = df['Length'].values.reshape(-1,1)
    y = df['MidrangePrice'].values

    # Specify the model.
    model = sklearn.linear_model.LinearRegression(fit_intercept=True)

    # Fit the data to the model.
    model.fit(X,y)

    # Extract the coeffecients.
    print("Bias (intercept) =",model.intercept_)
    print("Coefficients (slope) =", model.coef_[0])

    # Compute SSE and R-squared.
    predicted = model.predict(X)
    SSE = ((predicted - y)**2).sum()
    print("SSE = ", SSE)
    R_sq = model.score(X,y)
    print("R-squared = ", R_sq)

    # The predict function can be applied to a vector. Here we apply to it a
    # sequence of evenly-spaced values corresponding to our x-axis.
    xfit = np.linspace(140,230,600).reshape(-1,1)
    yfit = model.predict(xfit)

    # Plot the data
    plt.scatter(X,y)
    plt.title('1993 Car Data')
    plt.xlabel('Length')
    plt.ylabel('MidrangePrice ($10K)')
    plt.plot(xfit,yfit)
    plt.show()

def question2():
    # Assign to X and y. We have to reshape X to match
    # what the subsequent method expects.
    headers = ['CityMPG', 'HighwayMPG', 'EngineSize', 'Horsepower', 'RPM', 'EngineRevPerMile', 'ManualTransmission',
                   'FuelTankCapacity', 'PassengerCapacity', 'Length', 'Wheelbase', 'Width', 'UturnSpace', 'Weight',
                   'Domestic']
    best = 0
    for h in headers:
        if (h!='MinimumPrice' and h!='MaximumPrice'):
            X = df[h].values.reshape(-1,1)
            y = df['MidrangePrice'].values

            # Specify the model.
            model = sklearn.linear_model.LinearRegression(fit_intercept=True)

            # Fit the data to the model.
            model.fit(X,y)

            R_sq = model.score(X,y)
            if(R_sq > best):
                best = R_sq
                bestcriteria = h

    print('Best criteria that fits MidrangePrice, other than MinimumPrice and MaximumPrice = ',bestcriteria)

def question3():
    # Assign to X and y. We have to reshape X to match
    # what the subsequent method expects.
    headers = ['CityMPG', 'EngineSize', 'Horsepower', 'RPM', 'EngineRevPerMile', 'ManualTransmission',
                   'FuelTankCapacity', 'PassengerCapacity', 'Length', 'Wheelbase', 'Width', 'UturnSpace', 'Weight',
                   'Domestic']
    best = 0
    R_sq_list = []
    for h in headers:
        if (h!='CityMPG'):
            X = df[h].values.reshape(-1,1)
            y = df['HighwayMPG'].values

            # Specify the model.
            model = sklearn.linear_model.LinearRegression(fit_intercept=True)

            # Fit the data to the model.
            model.fit(X,y)

            R_sq = model.score(X,y)
            R_sq_list.append([R_sq,h])

    R_sq_list.sort(reverse=True)

    print('Best 3 criteria that fits HighwayMPG are = {}, {} and {}'.format((R_sq_list[0])[1], (R_sq_list[1])[1], (R_sq_list[2])[1]))

    headers = [(R_sq_list[0])[1], (R_sq_list[1])[1], (R_sq_list[2])[1]]

    for h in headers:
        X = df[h].values.reshape(-1, 1)
        y = df['MidrangePrice'].values

        # Specify the model.
        model = sklearn.linear_model.LinearRegression(fit_intercept=True)

        # Fit the data to the model.
        model.fit(X, y)

        print(h)
        # Extract the coeffecients.
        print("Bias (intercept) =", model.intercept_)
        print("Coefficients (slope) =", model.coef_[0])

        # Compute SSE and R-squared.
        predicted = model.predict(X)
        SSE = ((predicted - y) ** 2).sum()
        print("SSE = ", SSE)
        R_sq = model.score(X, y)
        print("R-squared = ", R_sq,"\n")

def question4to7():
    score = "0100001000100001000010001110101100010110000000100010000100001101100001110101000010000100000000011001"  # My score
    x = []
    y = []
    for i in range(0,100):
        x.append([i + 1])
        y.append(int(score[i]))

    # Specify the model.
    model = sklearn.linear_model.LinearRegression(fit_intercept=True)

    # Fit the data to the model.
    model.fit(x, y)

    # Find test statistic (Slope)
    test_statistic = model.coef_[0]
    print("Test Statistic (Slope) = ", test_statistic)

    test_stat_list = []  # List to store the test statistic for 10000 randomized permutations
    iterations = 10000

    for j in range(0, iterations):
        temp = list(score)
        random.shuffle(temp)  # Shuffling randomly

        x = []
        y = []
        for i in range(0, 100):
            x.append([i + 1])
            y.append(int(temp[i]))

        # Specify the model.
        model = sklearn.linear_model.LinearRegression(fit_intercept=True)

        # Fit the data to the model.
        model.fit(x, y)

        # Find test statistic (Slope)
        temp_stat = model.coef_[0]

        test_stat_list.append(temp_stat)  # Appending current test statistic

    hist(test_stat_list)

    test_stat_list.sort()  # Sorting in ascending
    index = np.searchsorted(test_stat_list, test_statistic)  # Finding index corresponding to our test statistic
    pvalue = 1.0 - (index + 1) / iterations

    alpha = 0.05
    confidence = 1 - alpha
    cutoff = test_stat_list[int(confidence * iterations)]

    print('P value = ', pvalue)
    print('Alpha value = ', alpha)
    print('Cutoff = ', cutoff)

    if (test_statistic > cutoff):
        print('We reject the null hypothesis. I improved.')
    else:
        print('We cannot reject the null hypothesis. I did not improve.')

def hist(x):
    plt.hist(x, density = True, bins = 50)
    plt.xlabel('Slope (Test Statistic)')
    plt.ylabel('Frequency')
    plt.show()

#question1()
#question2()
#question3()
question4to7()
