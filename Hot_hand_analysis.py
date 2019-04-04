import random
import numpy

score = "0100001000100001000010001110101100010110000000100010000100001101100001110101000010000100000000011001" # My score

ProbHH = 0 # Probability of a Hit after a Hit
ProbMH = 0 # Probability of a Hit after a Miss
ProbHM = 0 # Probability of a Miss after a Hit
ProbMM = 0 # Probability of a Miss after a Miss
for i in range(0, len(score) - 1): # Counting the number of Hit-Hits, Miss-Hits, Hits-Miss, Miss-Miss
    if score[i] == '1' and score[i + 1] == '1':
        ProbHH = ProbHH + 1
    if score[i] == '0' and score[i + 1] == '1':
        ProbMH = ProbMH + 1
    if score[i] == '1' and score[i + 1] == '0':
        ProbHM = ProbHM + 1
    if score[i] == '0' and score[i + 1] == '0':
        ProbMM = ProbMM + 1

# Calculating the probabilities of Hit-Hits and Miss-Hits for 100 shots
ProbHH = ProbHH / (ProbHH + ProbHM)
ProbMH = ProbMH / (ProbMH + ProbMM)
TestStatistic = ProbHH - ProbMH

TestStatList = [] # List to store the test statistic for 10000 randomized permutations
iterations = 10000

for j in range(0,iterations):
    temp = list(score)
    random.shuffle(temp) # Shuffling randomly
    temp = ''.join(temp)
    ProbRandHH = 0
    ProbRandMH = 0
    ProbRandHM = 0
    ProbRandMM = 0
    for i in range(0, len(temp) - 1):  # Counting the number of Hit-Hits, Miss-Hits, Hits-Miss, Miss-Miss
        if temp[i] == '1' and temp[i + 1] == '1':
            ProbRandHH = ProbRandHH + 1
        if temp[i] == '0' and temp[i + 1] == '1':
            ProbRandMH = ProbRandMH + 1
        if temp[i] == '1' and temp[i + 1] == '0':
            ProbRandHM = ProbRandHM + 1
        if temp[i] == '0' and temp[i + 1] == '0':
            ProbRandMM = ProbRandMM + 1

    ProbRandHH = ProbRandHH / (ProbRandHH + ProbRandHM)
    ProbRandMH = ProbRandMH / (ProbRandMH + ProbRandMM)
    TempStat = ProbRandHH - ProbRandMH
    TestStatList.append(TempStat) # Appending current test statistic

TestStatList.sort() # Sorting in ascending
index = numpy.searchsorted (TestStatList, TestStatistic) # Finding index corresponding to our test statistic
pvalue = 1.0 - (index + 1) / iterations

alpha = 0.05
confidence = 1-alpha
cutoff = TestStatList[int(confidence*iterations)]

print('Probability of a Hit after a Hit = ',ProbHH)
print('Probability of a Hit after a Miss = ',ProbMH)
print('Test Statistic = ', TestStatistic)
print('P value = ', pvalue)
print('Cutoff = ', cutoff)

if(TestStatistic > cutoff):
    print('We reject the null hypothesis. I did have a hot hand.')
else:
    print('We cannot reject the null hypothesis. I did not have a hot hand.')
