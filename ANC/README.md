適応ノイズキャンセラのプログラムです．<br>
・AdapriveNoiseCanceller.c<br>
  適応ノイズキャンセラの実機用プロジェクトです．<br>
  Adaptive_Noise_Canceller.cが適応ノイズキャンセラの処理になっています．<br>
	プログラム内の	<br>
	ch1 = *(volatile short *)DSKIF_AD2;// Reference signalが参照マイクロホンの入力チャネル<br>
	ch2 = *(volatile short *)DSKIF_AD1;// Primary signalが主マイクロホンの入力チャネル<br>
	*(volatile short *)DSKIF_DA3 = 5 * 3276.8 * error;が誤差信号の出力先となっています．<br>
 	主マイクロホンで雑音の重畳した音声，参照マイクロホンで騒音を取得し，適応フィルタにより，音声に重畳した雑音を低減します．<br>
 	誤差信号が，騒音低減後の出力信号となっています．<br>
  
・AdapriveNoiseCanceller_VSSNLMS.c<br>
	適応ノイズキャンセラの実機用プロジェクトです．<br>
 	適応フィルタの更新アルゴリズムとしてVSSNLMSを選択可能にしています．<br>
  	また，取得信号の高域強調や処理遅延に対応するための，信号への遅延付与にも対応しています．<br>

-------------------------------------------------------------------------------------------------------------------
使用OS : Windows XP<br>
開発環境 : Code Composer Studio<br>
使用DSP : TMS320C6713<br>
