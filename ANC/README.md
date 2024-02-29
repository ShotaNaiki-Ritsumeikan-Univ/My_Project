適応ノイズキャンセラのプログラムです．<br>
・AdapriveNoiseCanceller.c
  適応ノイズキャンセラの実機用プロジェクトです．
  Adaptive_Noise_Canceller.cが適応ノイズキャンセラの処理になっています．
	プログラム内の	
	ch1 = *(volatile short *)DSKIF_AD2;// Reference signalが参照マイクロホンの入力チャネル
	ch2 = *(volatile short *)DSKIF_AD1;// Primary signalが主マイクロホンの入力チャネル
	*(volatile short *)DSKIF_DA3 = 5 * 3276.8 * error;が誤差信号の出力先となっています．
 主マイクロホンで雑音の重畳した音声，参照マイクロホンで騒音を取得し，適応フィルタにより，音声に重畳した雑音を低減します．
 誤差信号が，騒音低減後の出力信号となっています．
