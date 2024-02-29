from scipy.io.wavfile import read,write
import numpy as np

def main():
    numSeed = 3462
    numElevation = 11
    numAzimuth = 13
    intangle = 15
    numInputChannels = 32
    time = 1
    Input = []
    f = open('temp.txt','w')

    #入力信号の作成
    for seed in range(numSeed):
        FS, WhiteNoise = read('sound/WhiteNoise_'+str(time)+'_'+str(seed+1)+'seed.wav')
        if (seed % 100) == 0:
            print('seed = '+str(seed))
            f.write('seed = '+str(seed)+'\n')
        
        for i in range(numElevation):
            for j in range(numAzimuth):

                if(seed % 100) == 0:
                    print('Elevation = '+str((i-4)*intangle))
                    print('Azimuth = '+str(j*intangle))
                
                #例外処理
                if i == numElevation-1:
                    if j != 0:
                        continue
                if i <= 2:
                    if j > 3:
                        continue
                
                #各チャネルごとに入力信号を作成
                for k in range(numInputChannels):
                    ReadFileName = 'WinImpulse_wav/rec_E' + str((i-4)*intangle) + 'A' + str(j*intangle) + '_' + str(k+1) + '.wav'
                    OutFileName = 'Input_wav/seed'+str(seed+1)+'/rec_E' + str((i-4)*intangle) + 'A' + str(j*intangle) + '/' + str(k+1) + '.wav'
                    FS, Impulse = read(ReadFileName)
                    temp = np.convolve(WhiteNoise, Impulse)
                    tempLen = len(temp)
                    Input.append(temp)
                    write(OutFileName, rate = FS, data = temp)
    
    #出力信号の作成
    Elevation = [-60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
    Azimuth = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    MatchingChannel = [
        [19, 19, 18, 18, 18, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1],
        [13, 13, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9],
        [17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,17, 17]
    ]

    for seed in range(numSeed):
        for i in range(numElevation):
            for j in range(numAzimuth):
                #例外処理
                if i == numElevation-1:
                    if j != 0:
                        continue
                if i <= 2:
                    if j > 3:
                        continue
        
            #音源方向に対応するレイヤーの算出
            if i <= 2:
                MatchingLayer = 0 #under
            elif 2 < i and i <= 5:
                MatchingLayer = 1 #mid
            elif 5 < i and i <= 8:
                MatchingLayer = 2 #Upper
            else:
                MatchingLayer = 3 #top

            #教師信号の生成
            OutFileName_train = 'Train_wav/seed' + str(seed+1) + '/rec_E' + str((i-4)*intangle) + 'A' + str(j*intangle) + '/' + str(k+1) + '.wav'
            for k in range(numInputChannels):
                if k != MatchingChannel[MatchingLayer][i]:
                    Input[k][:] = 0.0
                write(OutFileName_train,rate = FS, data = Input[k])


if __name__ == '__main__':
    main()
    print('END')
