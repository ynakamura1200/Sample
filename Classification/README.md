# 説明
Classificationは、機械学習（分類）に関するアプリです。  

正解ラベルのあるcsvファイルを元に、ディープラーニングを用いて学習を行います。  
学習中のlossの推移（損失関数）を、DBに保存した後にグラフとして可視化します。  

以下の図は、サンプルデータ（app/sample_files/sample_wine.csv）を用いた場合のグラフの例です。
この例では、学習係数を変えることによる結果の違いが確認できました。
![exampleResult](https://user-images.githubusercontent.com/58759616/224544745-247f46ac-da9a-476b-965e-5ed3379b79e4.png)

csvファイル入力画面の表示内容、機能は改修中です。 


# 使用している主な技術  
 【言語】  
　　Python  
 【ライブラリ】  
　　NumPy  
　　Pandas  
　　Dash  
　　SQLAlchemy  
　　FLASK  

# インストールしたライブラリ  
　dash1.8.0  
　dash-core-components1.7.0  
　dash-html-components1.0.2  
　dash-renderer1.2.3  
　dash-table4.6.0  
　Flask1.1.1  
　gunicorn19.7.1  
　numpy1.17.4  
　pandas0.25.3  
　plotly4.5.0  
　psycopg22.8.4  
　requests2.22.0  
　SQLAlchemy1.3.13  
