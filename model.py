import sys
#!{sys.executable} -m pip install tensorflow


#%% 分析モデル本体　==========================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. データ読み込みと前処理（差分の作成）
df = pd.read_csv('./data/ETTh1.csv')
df['OT_lag6'] = df['OT'].shift(6)
# 日付型に変換
df['date'] = pd.to_datetime(df['date'])


# 1. 曜日（0:月, 6:日）
df['day_of_week'] = df['date'].dt.dayofweek
# 2. 時間（0-23）
df['hour'] = df['date'].dt.hour

# 3. 月(month)と日(day)の抽出
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear # 1年の中の何日目か


# これまでの周期特徴量と合わせる
features = [
    'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL',
    'month',
    'hour'          # 時間
]


target_col = 'OT'


# ターゲットを「現在の温度 - 6時間前の温度」に変換
# 過去6時間で温度がどれだけ上昇/下降したかの「変化量」を学習させる
df['target_diff'] = (df[target_col] - df[target_col].shift(6))

# 差分計算で生じる最初の6行のNaNを除去
df = df.dropna()  # 差分計算で生じる末尾のNaNを除去

# スケーリング
scaler_X = StandardScaler()
scaler_y = StandardScaler()


X = scaler_X.fit_transform(df[features])
y = scaler_y.fit_transform(df[['target_diff']])


# 2. シーケンスデータの生成
time_steps =48

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X, y, time_steps)

# 3. LSTMモデル構築
model = Sequential([
    LSTM(256, activation='tanh', return_sequences=True, input_shape=(time_steps, len(features))),
    Dropout(0.2), 
    LSTM(128, activation='tanh'),
    Dropout(0.3), # ここにも追加
  #  Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
# 頂点付近の誤差も適切に評価する
#model.compile(optimizer='adam', loss='huber')

from tensorflow.keras.optimizers import Adam

# 学習率を 0.0005 に下げて、じっくり学習させる
optimizer = Adam(learning_rate=0.00005)
#model.compile(optimizer=optimizer, loss='huber')


# 4. 学習
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[early_stop] # これを追加
)




#%%　実測データと予測データの比較（グラフ）==========================================

import matplotlib.pyplot as plt

# 2. 損失関数の推移をグラフ化
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Convergence')
plt.legend()
plt.show()

# 3. 予測と逆スケーリング (元に戻す)
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
#y_pred = y_pred_scaled
y_test_inv = scaler_y.inverse_transform(y_test)
#y_test_inv = y_test

# 4. 波形の可視化 (最後の200データ分を比較)
import matplotlib.pyplot as plt

# フォントサイズの設定
fs = 17
# 4. 波形の可視化 (直近500データ分)
plt.figure(figsize=(16, 8)) # 文字を大きくするため少し高さを出しました
# プロット
plt.plot(y_test_inv[-500:], label='Actual', color='blue', alpha=0.6, linewidth=2)
plt.plot(y_pred[-500:], label='LSTM Prediction', color='red', linestyle='--', linewidth=2)
# タイトルと軸ラベル
plt.title('Actual vs LSTM Prediction', fontsize=fs + 3, fontweight='bold')
plt.xlabel('Time Steps (Hours)', fontsize=fs)
plt.ylabel('Temperature Change (°C)', fontsize=fs)
# 目盛り数字のサイズ
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
# 凡例のサイズと位置
plt.legend(fontsize=fs, loc='upper right')
# グリッド
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('prediction_comparison_large.png', dpi=300)
plt.show()

print("[Save] prediction_comparison_large.png を保存しました。")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 評価指標の計算
mae = mean_absolute_error(y_test_inv, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
r2 = r2_score(y_test_inv, y_pred)

print(f"MAE: {mae:.4f} (平均誤差)")
print(f"RMSE: {rmse:.4f} (大きな外れ値へのペナルティが強い指標)")
print(f"R2 Score: {r2:.4f} (モデルの当てはまりの良さ: 1に近いほど良い)")




#%%散布図. ==========================================

import matplotlib.pyplot as plt

# フォントサイズの設定（大きめ）
fs = 18

plt.figure(figsize=(10, 10)) # 正方形の散布図にするためサイズを調整

# 散布図プロット
plt.scatter(y_test_inv, y_pred, alpha=0.3, color='purple', s=40) # sで点のサイズを調整

# 対角線（完全一致のライン）を描画
min_val = min(y_test_inv.min(), y_pred.min())
max_val = max(y_test_inv.max(), y_pred.max())
# プロット範囲も少し広げる
buffer = (max_val - min_val) * 0.05
min_plot = min_val - buffer
max_plot = max_val + buffer

plt.plot([min_plot, max_plot], [min_plot, max_plot], color='red', linestyle='--', label='Perfect Fit', linewidth=2.5)

# タイトルと軸ラベル
plt.title('Actual vs Predicted: Scatter Plot', fontsize=fs + 4, fontweight='bold')
plt.xlabel('Actual Temperature Change (°C)', fontsize=fs) # 差分予測なのでラベルを変更
plt.ylabel('Predicted Temperature Change (°C)', fontsize=fs) # 差分予測なのでラベルを変更

# 目盛り数字のサイズ
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

# 凡例のサイズ
plt.legend(fontsize=fs)

# グリッド
plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
# 画像を保存（プレゼン用）
plt.savefig('scatter_actual_vs_predicted_large.png', dpi=300)
plt.show()

print("[Save] scatter_actual_vs_predicted_large.png を保存しました。")






# %%　解釈性 ==========================================

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import japanize_matplotlib

# フォントサイズの設定
fs = 17

# 0. 正しい特徴量名の定義
correct_feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'month', 'hour']

# 1. 勾配ベースの寄与度計算
def get_temporal_contribution(model, input_seq):
    input_tensor = tf.convert_to_tensor(input_seq, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        prediction = model(input_tensor)
    grads = tape.gradient(prediction, input_tensor)
    # 入力値と勾配の積で寄与度を近似 (Input x Gradient)
    return (input_seq * grads.numpy())

# 2. 寄与度の計算と平均化
num_samples = 100
sample_data = X_test[:num_samples]
contribs = get_temporal_contribution(model, sample_data) 
avg_contribution = np.mean(contribs, axis=0) 

# 3. 物理的直感に合わせた反転 (現在を左端に)
# 240ステップ（40時間分）のうち、直近の挙動を見やすくするため
flipped_contribution = np.flip(avg_contribution, axis=0)

# 4. 可視化
plt.figure(figsize=(16, 9))

# 寄与度の絶対値の最大値を取得してカラーバーの範囲を対称にする
vmax = np.max(np.abs(flipped_contribution))
vmin = -vmax

# ヒートマップ描画
img = plt.imshow(flipped_contribution.T, aspect='auto', cmap='RdBu_r', 
                 vmin=vmin, vmax=vmax, interpolation='nearest')

# カラーバーの設定
cbar = plt.colorbar(img)
cbar.set_label('Avg Contribution to Temperature Change', fontsize=fs)
cbar.ax.tick_params(labelsize=fs)

# 縦軸：特徴量名
plt.yticks(ticks=range(len(correct_feature_names)), labels=correct_feature_names, fontsize=fs)

# 横軸：0(現在)〜40(過去の時間)
# 10分刻みデータで240ステップ＝40時間分の場合
ticks = [0, 10,20,30,39]
labels = [0, 10, 20, 30, 40]
plt.xlim(0,40)
plt.xticks(ticks=ticks, labels=labels, fontsize=fs)

# 軸ラベルとタイトル
# 軸ラベルとタイトルの日本語化
plt.xlabel("予測時点からの経過時間 (ラグ/時間)", fontsize=fs)
plt.ylabel("入力特徴量", fontsize=fs)
plt.title("モデルの推論根拠マップ：入力寄与度の可視化", fontsize=fs + 4, fontweight='bold')

plt.tight_layout()
# 保存用
plt.savefig('model_interpretability_map.png', dpi=300)
plt.show()

print("[Save] model_interpretability_map.png を保存しました。")





