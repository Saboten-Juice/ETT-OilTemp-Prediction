#%%

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. データの読み込み (例として1時間おきのETTh1を使用)
file_path = "data/ETTh1.csv"
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    exit()

df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

print("--- データセット概要 ---")
print(df.info())
print("\n--- 統計量 ---")
print(df.describe())

# 2. 油温 (OT) の時系列推移の可視化
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['OT'], color='red', linewidth=1)
plt.title('油温 (OT) の全期間推移', fontsize=15)
plt.xlabel('日付')
plt.ylabel('温度 (℃)')
plt.grid(True, alpha=0.3)
plt.savefig('eda_ot_trend.png')
print("\n[Save] eda_ot_trend.png を保存しました。")



# 3. 各変数間の相関関係 (ヒートマップ)
import matplotlib.pyplot as plt
import seaborn as sns

# フォントサイズの設定
fs = 17

plt.figure(figsize=(13, 11)) # 文字サイズに合わせて少し余裕を持たせる

# cbar_kws でカラーバー自体の設定を調整可能
ax = sns.heatmap(df.corr(), annot=True, annot_kws={'size': fs}, 
                 cmap='RdBu_r', center=0)

# 1. タイトル
plt.title('変数間の相関係数', fontsize=fs + 3)

# 2. 軸ラベル（変数名）
ax.tick_params(axis='both', labelsize=fs)

# 3. カラーバーの目盛り数字のサイズを変更
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=fs)

plt.savefig('eda_correlation_final.png', bbox_inches='tight')
plt.show()
print("[Save] eda_correlation_final.png を保存しました（カラーバーも拡大）。")




# 4. 特定期間の拡大 (全負荷系統と温度の関係を見る)
# 最初の1週間 (24時間 * 7日 = 168サンプル) を表示
sample_df = df.head(168)
# フォントサイズを一括管理するための変数
fs = 16 

fig, ax1 = plt.subplots(figsize=(18, 9)) # 16ptだと文字が大きくなるため、少し図自体のサイズを広げました

ax1.set_xlabel('日付', fontsize=fs)
ax1.set_ylabel('負荷 (Load Currents)', color='black', fontsize=fs)

# 各系統プロット
ax1.plot(sample_df.index, sample_df['HUFL'], label='High Full (HUFL)', color='royalblue', alpha=0.7)
ax1.plot(sample_df.index, sample_df['HULL'], label='High Least (HULL)', color='lightskyblue', alpha=0.5, linestyle='--')
ax1.plot(sample_df.index, sample_df['MUFL'], label='Mid Full (MUFL)', color='forestgreen', alpha=0.7)
ax1.plot(sample_df.index, sample_df['MULL'], label='Mid Least (MULL)', color='limegreen', alpha=0.5, linestyle='--')
ax1.plot(sample_df.index, sample_df['LUFL'], label='Low Full (LUFL)', color='darkorchid', alpha=0.7)
ax1.plot(sample_df.index, sample_df['LULL'], label='Low Least (LULL)', color='plum', alpha=0.5, linestyle='--')

# 左軸の目盛り
ax1.tick_params(axis='both', labelsize=fs)
# 凡例の文字サイズ
ax1.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=fs)

# 右軸に温度を表示
ax2 = ax1.twinx()
ax2.set_ylabel('油温 (OT)', color='tab:red', fontsize=fs, fontweight='bold')
ax2.plot(sample_df.index, sample_df['OT'], color='tab:red', linewidth=3, label='油温 (OT)')
# 右軸の目盛り
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=fs)

plt.title('最初の1週間の全負荷系統と油温の推移', fontsize=fs + 4) # タイトルは少し大きめに設定
fig.tight_layout()
plt.savefig('eda_weekly_detail.png')
plt.show() # 確認用に追加
print("[Save] eda_weekly_detail.png を更新しました。")


# 5. 各負荷項目とOTの相関を数値で確認
print("\n--- 油温(OT)との相関係数 ---")
correlations = df.corr()['OT'].sort_values(ascending=False)
print(correlations)

# 6. 各負荷系統ごとの個別推移と油温の関係 (3x2サブプロット)
fig, axes = plt.subplots(3, 2, figsize=(18, 15), sharex=True)
axes = axes.flatten()
load_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
colors = ['royalblue', 'lightskyblue', 'forestgreen', 'limegreen', 'darkorchid', 'plum']

for i, col in enumerate(load_cols):
    ax1 = axes[i]
    ax1.plot(sample_df.index, sample_df[col], color=colors[i], label=col)
    ax1.set_ylabel(f'負荷 ({col})', color=colors[i])
    ax1.tick_params(axis='y', labelcolor=colors[i])
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(sample_df.index, sample_df['OT'], color='tab:red', alpha=0.5, label='OT')
    ax2.set_ylabel('油温 (OT)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    ax1.set_title(f'{col} と 油温 (OT) の比較')
    ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_loads_individual.png')
print("[Save] eda_loads_individual.png を保存しました。")

# 7. 負荷の分布比較 (箱ひげ図)
plt.figure(figsize=(12, 6))
df[load_cols].boxplot()
plt.title('各負荷系統の分布比較', fontsize=15)
plt.ylabel('負荷値')
plt.grid(True, alpha=0.3)
plt.savefig('eda_loads_boxplot.png')
print("[Save] eda_loads_boxplot.png を保存しました。")

print("\nEDA完了。保存された画像を確認して、データの傾向を分析してください。")


# %%　３日分の油温グラフ
