import streamlit as st 
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from PIL import Image
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import base64
from io import BytesIO

#タイトルの記入
st.title("データ予測アプリ")

st.header('CSVファイルについて')
st.write('CSVの1列目に予測したい値を入力し、2列目以降に予測に必要な値を入力してください。')
st.write('CSVの1列目の最終行に予測したい内容を1列目のみ空欄で追加していってください')
#im = Image.open("IMG.jpg")
#width = 800
#ratio = width / im.width
#height = int(im.height * ratio) #(5)
#im_resized = im.resize((width, height))
#im_resized.save('img1.jpg')
st.image("IMG.JPG")

#ファイルのアップロード(CSV)他にも画像や音声や動画もOK
uploaded_file = st.file_uploader("ファイルの取り込み", type='csv')

#ファイルがアップロードされてからの処理
if uploaded_file is not None:
    df =pd.read_csv(uploaded_file ,encoding="SHIFT-JIS")
    
    #相関の確認
    if st.button('相関関係を確認'):
        comment = st.empty()
        comment.write('相関確認を確認してます。少々お待ちください。')

        df1 = df.corr()
        st.dataframe(df1)
        
        comment.write('相関確認完了')
       
    #分析の実施
    if st.button('予測を開始'):
    
        comment = st.empty()
        comment.write('分析を開始してます。少々お待ちください。')

        #実際予測に利用する説明変数をX2に代入
        X2 = df[df.iloc[:,:1].isnull().any(axis=1)]
        X2 =X2.iloc[:,1:]

        #欠損行の除去
        df = df.dropna()

        #価格をy、それ以外をxに代入
        y = df.iloc[:,:1]
        x = df.iloc[:,1:]

        #訓練データとテストデータに分割
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

        #重回帰分析
        model=LinearRegression()
        model.fit(x_train,y_train)
        A1 = model.score(x_test,y_test)

        #サポートベクターマシーン
        model2 = SVR(gamma='auto')
        model2.fit(x_train, y_train)
        A2 = model2.score(x_test,y_test)

        #ランダムフォレスト
        model3 = RandomForestRegressor(max_depth=10,n_estimators=10)
        model3.fit(x_train, y_train)
        A3 = model3.score(x_test,y_test)

        #各アルゴリズムでスコアが高いものを採用
        if A1 > A2 and A1 > A3:
            Y_pred = model.predict(X2)
            A1 = str(math.ceil(A1*100))
            name = "重回帰分析で予測完了しました。精度は" + A1 + "%です。"
        elif A2 > A1 and A2 > A3:
            Y_pred = model2.predict(X2)
            A2 = str(math.ceil(A2*100))
            name = "サポートベクターマシーンで予測完了しました。精度は" + A2 + "%です。"
        else:
            Y_pred = model3.predict(X2)
            A3 = str(math.ceil(A3*100))
            name = "ランダムフォレストで予測完了しました。精度は" + A3 + "%です。"

        #結果をリスト型に格納    
        Y_pred = Y_pred.tolist()
        #小数点は第二まで
        Y_pred = [round(Y_pred[n], 2) for n in range(len(Y_pred))]
        
        #予測結果を追記
        X2['Predict'] = Y_pred
        X3 = X2
        
        #CSVダウンロード機能の追加
        csv = X3.to_csv(index=False)
        b64 = base64.b64encode(csv.encode("SHIFT-JIS")).decode()  # some strings
        linko= f'<a href="data:file/csv;base64,{b64}" download="result.csv">Download csv file</a>'
        st.markdown(linko, unsafe_allow_html=True)
        
        #結果を表示
        X2 = X2.style.set_properties(**{"background-color":"orange"}, subset=["Predict"])
        st.dataframe(X2)
        comment.write(name)
