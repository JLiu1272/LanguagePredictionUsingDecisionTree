# Library for loading a csv file and converting
# it into a dictionary
import pandas as pd
import csv
def decisionTree_predict(csv_file):
	datas = pd.read_csv(csv_file)
	TP = 0
	TN = 0
	with open('Lab_02_Liu_Jennifer_ResultClassifications.csv', 'w') as result_csv:
		writer = csv.writer(result_csv)
		writer.writerow(['result'])
		for data_record in range(datas.shape[0]):
			if datas.iloc[data_record]['VowelCombDutch'] <= 0:
			  if datas.iloc[data_record]['CommonDutch'] <= 0:
			    Lang = 0
			  else:
			    Lang = 0
			else:
			  if datas.iloc[data_record]['StopwordsEng'] <= 0:
			    Lang = 0
			  else:
			    Lang = 1
			result = 'en' if Lang == 1 else 'nl'
			print(result)
			writer.writerow([Lang])
			if Lang == datas.iloc[data_record]['Lang']:
				if Lang == 0:
					TN += 1
				else:
					TP += 1

		result_csv.close()
		print('TP: ' + str(TP) + ' TN: ' + str(TN))
		accuracy = (TP + TN)/datas.shape[0]
		print('Accuracy: ' + str(accuracy))

decisionTree_predict("processed_data/binary_dutch_eng_validation.csv")