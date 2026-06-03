import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────
N, K, LOOKBACK = 20, 2, 10

ZONE_COLORS = {-2:'#FF4C4C', -1:'#FFA500', 1:'#00BFFF', 2:'#00E676'}
DARK_BG, GRID_COL, TEXT_COL, ACCENT = '#0D1117', '#1E2A38', '#E6EDF3', '#58A6FF'

FEATURE_COLS = ['open', 'close', 'momentum', 'volatility',
                'band_width', 'close_to_mean', 'close_pct_band']


# ── Data Loading & Feature Computation ────────────────────────
def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={'bb up band':'upper_band','bb mid':'middle_band','bb down band':'lower_band'})

    missing = {'open','high','low','close','mean','sd'} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Bollinger Bands
    df['middle_band'] = df['mean']
    df['upper_band']  = df['mean'] + K * df['sd']
    df['lower_band']  = df['mean'] - K * df['sd']

    # Momentum & Volatility
    df['momentum']   = df['close'].diff(N)
    df['volatility'] = df['close'].rolling(N).std()

    # Derived band features
    df['band_width']     = df['upper_band'] - df['lower_band']
    df['close_to_mean']  = df['close'] - df['mean']
    df['close_pct_band'] = (df['close'] - df['lower_band']) / (df['band_width'] + 1e-9)

    df = df.dropna(subset=['momentum','volatility']).reset_index(drop=True)
    df['x'] = np.arange(1, len(df) + 1)

    # Zone labels & positions
    conds   = [df['close']>df['upper_band'],
               (df['close']<=df['upper_band'])&(df['close']>df['middle_band']),
               (df['close']<=df['middle_band'])&(df['close']>df['lower_band']),
               df['close']<=df['lower_band']]
    df['zone_label'] = np.select(conds, [-2,-1,1,2], default=0)
    df['position']   = df['zone_label'].map({2:1,1:0,-1:0,-2:-1}).fillna(0).astype(int)
    return df


# ── Rolling Window Features ────────────────────────────────────
def build_rolling_features(df: pd.DataFrame, lookback: int = LOOKBACK):
    vals, targets = df[FEATURE_COLS].values, df['zone_label'].values
    X, y, idx = [], [], []
    for t in range(lookback, len(df)):
        X.append(vals[t-lookback:t].flatten())
        y.append(targets[t])
        idx.append(t)
    col_names = [f"{f}_lag{l}" for l in range(lookback,0,-1) for f in FEATURE_COLS]
    return np.array(X), np.array(y), np.array(idx), col_names


# ── Split & Scale ──────────────────────────────────────────────
def split_and_scale(X, y, train_ratio=0.8):
    s = int(len(X) * train_ratio)
    Xtr, Xte, ytr, yte = X[:s], X[s:], y[:s], y[s:]
    sc_tr, sc_te = StandardScaler(), StandardScaler()
    print(f"\n  Train: {len(Xtr)}  |  Test: {len(Xte)}")
    return sc_tr.fit_transform(Xtr), sc_te.fit_transform(Xte), ytr, yte, sc_tr, sc_te, s


# ── Train & Evaluate ───────────────────────────────────────────
def train_model(Xtr, ytr, Xte, yte, col_names):
    model = LinearRegression().fit(Xtr, ytr)
    coefs = dict(zip(col_names, model.coef_))
    coefs['intercept'] = model.intercept_

    print("\n" + "═"*55)
    print("  FEATURE GROUP IMPORTANCE  (mean |coef| over lags)")
    print("═"*55)
    for feat in FEATURE_COLS:
        vals = [abs(v) for k,v in coefs.items() if k.startswith(feat)]
        avg  = np.mean(vals)
        print(f"  {feat:<18} {avg:.6f}  {'█'*min(int(avg*15),30)}")
    print("═"*55)
    for lbl, X, y in [('TRAIN',Xtr,ytr),('TEST',Xte,yte)]:
        p = model.predict(X)
        print(f"  {lbl}  R²={r2_score(y,p):.4f}  MSE={mean_squared_error(y,p):.4f}")
    print("═"*55 + "\n")
    return model, coefs


# ── Inference ─────────────────────────────────────────────────
def run_inference(df, X, indices, model, sc_tr, sc_te, split_idx, lookback=LOOKBACK):
    df = df.copy()
    df['predicted_zone'] = np.nan
    df['split'] = 'train'
    for i, (row, ridx) in enumerate(zip(X, indices)):
        sc = sc_tr if i < split_idx else sc_te
        df.at[ridx,'split'] = 'train' if i < split_idx else 'test'
        df.at[ridx,'predicted_zone'] = int(np.round(model.predict(sc.transform(row.reshape(1,-1)))[0]).clip(-2,2))
    return df


# ── Plotting ───────────────────────────────────────────────────
def _style(ax, title):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COL); ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(ACCENT)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle='--')
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)

def _shade_split(ax, x, df_w):
    for sp, col in [('train','#00BFFF'),('test','#FF4C4C')]:
        m = df_w['split']==sp
        if m.any(): ax.axvspan(x[m][0], x[m][-1], alpha=0.05, color=col, label=sp.title())

def _plot_window(df_w, coefs, start, end, path):
    x = df_w['x'].values
    fig = plt.figure(figsize=(24,30), facecolor=DARK_BG)
    fig.suptitle(f'Bollinger Bands ML Pipeline  |  n=20 k=2  |  Rows {start}–{end}',
                 fontsize=16, fontweight='bold', color=ACCENT, y=0.99)
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    # ① Bollinger Bands
    ax1 = fig.add_subplot(gs[0,:])
    _style(ax1, f'① Bollinger Bands [{start}–{end}]')
    ax1.plot(x, df_w['close'],       '#FFFFFF', lw=0.8, label='Close',       zorder=5)
    ax1.plot(x, df_w['upper_band'],  '#FF4C4C', lw=0.9, label='Upper',  linestyle='--')
    ax1.plot(x, df_w['middle_band'], '#FFA500', lw=0.9, label='Middle', linestyle='-.')
    ax1.plot(x, df_w['lower_band'],  '#00E676', lw=0.9, label='Lower',  linestyle='--')
    ax1.fill_between(x, df_w['lower_band'], df_w['upper_band'], alpha=0.07, color=ACCENT)
    _shade_split(ax1, x, df_w)
    ax1.set(xlabel='Minute Index', ylabel='Price')
    ax1.legend(fontsize=8, facecolor='#161B22', labelcolor=TEXT_COL, framealpha=0.8)

    # ② Momentum & Volatility
    ax2 = fig.add_subplot(gs[1,0]); _style(ax2, '② Momentum & Volatility')
    ax2t = ax2.twinx()
    ax2.plot(x, df_w['momentum'],   '#FFD700', lw=0.8, label='Momentum')
    ax2.axhline(0, color='#444C56', lw=0.6, linestyle='--')
    ax2t.plot(x, df_w['volatility'],'#DA70D6', lw=0.8, label='Volatility')
    ax2.set(xlabel='Minute Index', ylabel='Momentum')
    ax2t.set_ylabel('Volatility', color='#DA70D6')
    ax2t.tick_params(colors='#DA70D6', labelsize=8)
    lines = ax2.get_legend_handles_labels()[0] + ax2t.get_legend_handles_labels()[0]
    lbls  = ax2.get_legend_handles_labels()[1] + ax2t.get_legend_handles_labels()[1]
    ax2.legend(lines, lbls, fontsize=7, facecolor='#161B22', labelcolor=TEXT_COL)

    # ③ Position Signals
    ax3 = fig.add_subplot(gs[1,1]); _style(ax3, '③ Position Signals')
    ax3.axhline(0, color=GRID_COL, lw=1)
    for pos, col, mk, lbl, sz, al in [(1,'#00E676','^','BUY',40,1.0),(-1,'#FF4C4C','v','SELL',40,1.0),(0,'#888888','o','CLOSE',6,0.3)]:
        m = df_w['position']==pos
        ax3.scatter(x[m], df_w['position'].values[m], color=col, marker=mk, s=sz, label=lbl, zorder=5, alpha=al)
    ax3.set_yticks([-1,0,1]); ax3.set_yticklabels(['SELL','CLOSE','BUY'], color=TEXT_COL)
    ax3.set_xlabel('Minute Index')
    ax3.legend(fontsize=7, facecolor='#161B22', labelcolor=TEXT_COL)

    # ④ Actual vs Predicted
    ax4 = fig.add_subplot(gs[2,0]); _style(ax4, '④ Actual vs Predicted Zone Labels')
    pv = df_w.dropna(subset=['predicted_zone'])
    for sp, col in [('train','#00BFFF'),('test','#FF4C4C')]:
        seg = pv[pv['split']==sp]
        if not seg.empty:
            ax4.plot(seg['x'].values, seg['predicted_zone'].values, color=col, lw=0.8,
                     linestyle='--', label=f'Pred ({sp})', alpha=0.85)
    ax4.plot(x, df_w['zone_label'], '#FFFFFF', lw=0.7, label='Actual', alpha=0.9)
    ax4.fill_between(x, 0, df_w['zone_label'].values - df_w['predicted_zone'].fillna(0).values,
                     alpha=0.12, color='#FFA500', label='Residual')
    ax4.set(xlabel='Minute Index', ylabel='Zone Label')
    ax4.legend(fontsize=7, facecolor='#161B22', labelcolor=TEXT_COL)

    # ⑤ Top-20 Coefficients
    ax5 = fig.add_subplot(gs[2,1]); _style(ax5, '⑤ Top-20 Coefficients')
    items = sorted([(f,v) for f,v in coefs.items() if f!='intercept'], key=lambda t:abs(t[1]), reverse=True)[:20]
    items.sort(key=lambda t:t[1])
    feats, vals = zip(*items)
    bars = ax5.barh(feats, vals, color=['#00E676' if v>=0 else '#FF4C4C' for v in vals],
                    edgecolor='#0D1117', height=0.6)
    for bar,val in zip(bars,vals):
        ax5.text(val+(0.0005 if val>=0 else -0.0005), bar.get_y()+bar.get_height()/2,
                 f'{val:+.4f}', va='center', ha='left' if val>=0 else 'right', color=TEXT_COL, fontsize=6)
    ax5.axvline(0, color=TEXT_COL, lw=0.8)
    ax5.tick_params(axis='y', labelsize=6)
    ax5.set_xlabel('Coefficient Value')
    plt.show()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig); print(f"  ✅  Saved → {path}")

def _plot_signals(df_w, start, end, path):
    x = df_w['x'].values
    fig, ax = plt.subplots(figsize=(24,6), facecolor=DARK_BG)
    _style(ax, f'⑥ Close with Signals  [{start}–{end}]')
    ax.plot(x, df_w['close'],       '#FFFFFF', lw=0.7, label='Close', zorder=3)
    ax.plot(x, df_w['upper_band'],  '#FF4C4C', lw=0.5, linestyle='--', alpha=0.4, label='Upper')
    ax.plot(x, df_w['lower_band'],  '#00E676', lw=0.5, linestyle='--', alpha=0.4, label='Lower')
    ax.fill_between(x, df_w['lower_band'], df_w['upper_band'], alpha=0.05, color=ACCENT)
    _shade_split(ax, x, df_w)
    for pos, col, mk, lbl, sz, al in [(1,'#00E676','^','BUY',60,1.0),(-1,'#FF4C4C','v','SELL',60,1.0),(0,'#888888','o','CLOSE',6,0.3)]:
        m = (df_w['position']==pos).values
        ax.scatter(x[m], df_w['close'].values[m], marker=mk, color=col, s=sz, zorder=6, alpha=al, label=f"{lbl}({m.sum()})")
    ax.set(xlabel='Minute Index', ylabel='Price')
    ax.legend(fontsize=8, facecolor='#161B22', labelcolor=TEXT_COL, framealpha=0.9)
    plt.show() 
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig); print(f"  ✅  Saved → {path}")

def plot_all(df, coefs, save_path='Bollinger_band.png', window_size=5000):
    base, ext = save_path.rsplit('.', 1)
    paths = []
    for s in range(0, len(df), window_size):
        dw = df.iloc[s:s+window_size]
        xs, xe = int(dw['x'].iloc[0]), int(dw['x'].iloc[-1])
        _plot_window(dw, coefs, xs, xe, f"{base}_{xs}_{xe}.{ext}")
        _plot_signals(dw, xs, xe, f"{base}_{xs}_{xe}_signals.{ext}")
        paths += [f"{base}_{xs}_{xe}.{ext}", f"{base}_{xs}_{xe}_signals.{ext}"]
    return paths


# ── Master Pipeline ────────────────────────────────────────────
def run_pipeline(train_csv, new_csv=None, window_size=5000, lookback=LOOKBACK):
    print("\n" + "═"*55)
    print("  BOLLINGER BANDS ML PIPELINE  |  n=20  k=2  |  lookback=10")
    print("═"*55)

    print("\n[1/7] Loading & preparing training data...")
    train_df = load_and_prepare(train_csv)
    print(f"      Rows: {len(train_df)}")

    print(f"[2/7] Building {lookback}-candle rolling features...")
    X, y, idx, col_names = build_rolling_features(train_df, lookback)
    print(f"      Samples: {len(X)}  |  Features per sample: {X.shape[1]}")

    print("[3/7] Splitting 80:20 & scaling separately...")
    Xtr_s, Xte_s, ytr, yte, sc_tr, sc_te, split_idx = split_and_scale(X, y)

    print("[4/7] Training Linear Regression...")
    model, coefs = train_model(Xtr_s, ytr, Xte_s, yte, col_names)

    if new_csv:
        print("[5/7] Loading new data for inference...")
        inf_df = load_and_prepare(new_csv)
        Xi, yi, ii, _ = build_rolling_features(inf_df, lookback)
        df_inf = run_inference(inf_df, Xi, ii, model, sc_tr, sc_te, split_idx=0, lookback=lookback)
    else:
        print("[5/7] Running inference on training data...")
        df_inf = run_inference(train_df, X, idx, model, sc_tr, sc_te, split_idx, lookback)

    n_win = int(np.ceil(len(df_inf) / window_size))
    print(f"[6/7] Plotting ({n_win} window(s))...")
    saved = plot_all(df_inf, coefs, window_size=window_size)
    print(f"      {len(saved)} files saved.")

    print("[7/7] Summary")
    print(f"      Candles: {len(df_inf)}  |  BUY: {(df_inf['position']==1).sum()}"
          f"  |  SELL: {(df_inf['position']==-1).sum()}  |  CLOSE: {(df_inf['position']==0).sum()}")
    print(df_inf['zone_label'].value_counts().sort_index().to_string())
    print("\n  ✅  Pipeline complete!\n")

    return df_inf, model, sc_tr, sc_te, coefs


# ── Entry Point ────────────────────────────────────────────────
if __name__ == '__main__':
    run_pipeline(train_csv='/content/bb_data_s.csv.csv.csv')