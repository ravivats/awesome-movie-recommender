import zipfile
import time
from surprise import Reader,  Dataset, evaluate, print_perf
from surprise import SVD, SVDpp, NMF, KNNWithMeans, SlopeOne, CoClustering


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
	
	print("Choose an option:\n")
	print("1. Predict rating of a certain item for a certain user.\n2. Skip this option.\n")
	choice= int(input())
	if (choice == 1):
		userid = str(input("Enter the User ID: "))
		itemid = str(input("Enter the Item ID: "))
		# Retrieve the trainset.
		trainset = data.build_full_trainset()
		algo.train(trainset)

		# Predict a certain item, userid = str(196), itemid = str(302)
		actual_rating = 4
		print(algo.predict(userid, itemid, actual_rating))		
	
	print("Press 1 to continue, 0 to quit.\n")
	cont= int(input())


