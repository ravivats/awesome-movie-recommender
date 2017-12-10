from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import zipfile, time
from surprise import Reader,  Dataset, evaluate, print_perf
from surprise import SVD, SVDpp, NMF, KNNWithMeans, SlopeOne, CoClustering
from collections import  defaultdict

def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # Map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
	
	
# Unzip ml-100k.zip
zipfile = zipfile.ZipFile('ml-100k.zip', 'r')
zipfile.extractall()
zipfile.close()

# Read data into an array of strings
with open('./ml-100k/u.data') as f:
    all_lines = f.readlines()

# Prepare the data to be used in Surprise
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file('./ml-100k/u.data', reader=reader)
cont=1

while(cont):
	print("Choose an algorithm:\n")
	print("1. SVD  2. SVD++  3. NMF  4. KNNWithMeans  5. Slope one  6. Co-Clustering\n")
	choice = int(input())

	# Choose the algorithm 
	if (choice == 1):
		algo = SVD()
	elif (choice == 2):
		algo = SVDpp()
	elif (choice == 3):
		algo = NMF()
	elif (choice == 4):
		algo = KNNWithMeans()
	elif (choice == 5):
		algo = SlopeOne()
	elif (choice == 6):
		algo = CoClustering()
			
	# Split the dataset into 5 folds 
	data.split(n_folds=5)
		
	# Train and test reporting the RMSE and MAE scores
	# Record algorithm execution start time
	start_time = time.time()
	
	perf= evaluate(algo, data, measures=['RMSE', 'MAE'])
	print_perf(perf)
	
	# Record algorithm execution end time and calculate execution time
	print("Execution time: --- %s seconds ---" % (time.time() - start_time))
	
	# Retrieve the training set.
	trainset = data.build_full_trainset()
	algo.train(trainset)
	
	print("Choose an option:\n")
	print("1. Predict rating of a certain item for a certain user.\n2. Get top n movie recommendations for a certain user.  \n3.Skip this option.\n")
	choice= int(input())
	if (choice == 1):
		userid = str(input("Enter the User ID: "))
		itemid = str(input("Enter the Item ID: "))
		
		# Predict a certain item, userid = str(196), itemid = str(302)
		actual_rating = 4
		print(algo.predict(userid, itemid, actual_rating))	
		
	elif (choice  == 2):
		print("\nEnter the User IDs (seperated by spaces):\n")
		userlist = map(str, input().split())
		testset = trainset.build_anti_testset()
		predictions = algo.test(testset)
		
		top_n = get_top_n(predictions, n=10)
		# Print the recommended items for each user
		for uid in userlist:
			print(uid, [iid for (iid, _) in top_n.get(uid)])
		
	print("Press 1 to continue, 0 to quit.\n")
	cont= int(input())
	