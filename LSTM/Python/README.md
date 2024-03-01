PythonのPytorchおよびPytorch lightningを用いた長短期記憶ネットワーク(Long-short Term Memory : LSTM)の学習プログラムです．<br>
卒業研究時に22.2マルチチャネル音響の収録用マイクロホンの指向性制御の提案手法として実装しました．<br>
・MakeDataset.py<br>
  測定した22.2マルチチャネル音響の収録用マイクロホンの伝達特性から学習データおよび教師データを生成します．<br>
・MakeCSV.py<br>
  LSTMの学習の際に，学習データおよび教師データのアノテーションのために使用するCSVファイルを生成します．<br>
・Library.py<br>
  Pytorchを用いてLSTMのモデルを定義します．<br>
・main.py<br>
  Pytorchを用いたLSTMの学習プログラム<br>
  MakeDataset.pyで生成した学習データおよび教師データを基に指向性制御のモデル学習を行います.<br>
・main_lightning<br>
  LSTMの学習プログラムのPytorch Lightning版，複数のGPUでの並列学習に対応しています．<br>
  
--------------------------------------------------------------------------------------------------------------------
動作環境
使用OS : Linux
使用言語 : Python3
使用ライブラリ :lightning, torch
