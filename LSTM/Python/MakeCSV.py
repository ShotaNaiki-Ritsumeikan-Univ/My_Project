import csv
import pprint

numSeed = 4200
numElevation = 11
numAzimuth = 13
intangle = 15

with open('Validation_Input.csv', 'w') as F:
    for i in range(3500,numSeed):
        for j in range(numElevation):
            for k in range(numAzimuth):

                #例外処理
                if j == numElevation-1:
                    if k != 0:
                        continue
                if j <= 2:
                    if k > 3:
                        continue

                temp = 'Validation_Input_wav/seed'+str(i+1)+'/rec_E' + str((j-4) * intangle) + 'A' + str(k * intangle) + '/'
                writer = csv.writer(F)
                writer.writerow([temp])
