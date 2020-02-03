# 説明
Classificationは、機械学習（分類）に関するアプリです。  

正解ラベルのあるcsvファイルを元に、ディープラーニングを用いて学習を行います。  
学習中のlossの推移（損失関数）を、DBに保存した後にグラフとして可視化します。  

サンプルデータ（app/sample_files/sample_iris.csv）を用いた場合のグラフは、  
以下のURLで確認できます。  
https://ynakamura-app-plot.herokuapp.com/  
学習の条件は、現在のところ ほぼ固定値です。  

以下のURLがcsvファイルの入力画面ですが、表示内容、機能は改修中です。  
https://ynakamura-app.herokuapp.com/  


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
