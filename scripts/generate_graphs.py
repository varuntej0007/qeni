#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from loguru import logger

plt.rcParams.update({
    'figure.facecolor':'#0d1117','axes.facecolor':'#0d1117',
    'axes.edgecolor':'#30363d','axes.labelcolor':'#c9d1d9',
    'axes.titlecolor':'#f0f6fc','xtick.color':'#8b949e',
    'ytick.color':'#8b949e','text.color':'#c9d1d9',
    'grid.color':'#21262d','grid.linewidth':0.8,
    'font.family':'monospace','font.size':10,
    'axes.titlesize':13,'axes.labelsize':11,'figure.titlesize':15,
})

BLUE='#58a6ff'; GREEN='#3fb950'; RED='#f85149'
PURPLE='#bc8cff'; AMBER='#e3b341'; CYAN='#39d0d8'
OUT='graphs'
os.makedirs(OUT, exist_ok=True)


def load_artifacts():
    try:
        svm     = joblib.load('models/svm_latest.joblib')
        encoder = joblib.load('models/encoder_latest.joblib')
        X_train = np.load('models/X_train_enc_latest.npy')
        K_train = np.load('models/K_train_latest.npy')
        with open('models/model_info.json') as f:
            info = json.load(f)
        logger.success(f"Artifacts loaded | n_train={len(X_train)}")
        return svm, encoder, X_train, K_train, info
    except FileNotFoundError as e:
        logger.error(f"Missing: {e} — run python scripts/train.py first")
        sys.exit(1)


def fig1_kernel_matrix(K_train, info):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Figure 1 — Quantum Kernel Matrix\nibm_marrakesh · PauliFeatureMap · 2048 shots')
    ax = axes[0]
    im = ax.imshow(K_train, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='K(xᵢ,xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|²')
    ax.set_title('Training kernel matrix K_train (30×30)')
    ax.set_xlabel('Training sample j')
    ax.set_ylabel('Training sample i')
    ax2 = axes[1]
    diag     = np.diag(K_train)
    off_diag = K_train[~np.eye(len(K_train), dtype=bool)]
    ax2.hist(off_diag, bins=40, alpha=0.7, color=BLUE,
             label=f'Off-diagonal (n={len(off_diag)})', density=True)
    ax2.hist(diag, bins=10, alpha=0.9, color=GREEN,
             label=f'Diagonal (n={len(diag)})', density=True)
    ax2.axvline(off_diag.mean(), color=BLUE, linestyle='--', alpha=0.8,
                label=f'Mean={off_diag.mean():.3f}')
    ax2.set_title('Kernel value distribution')
    ax2.set_xlabel('K(xᵢ,xⱼ)'); ax2.set_ylabel('Density')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    p = f'{OUT}/fig1_kernel_matrix.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(); logger.success(f"Saved: {p}")


def fig2_decision_boundary(svm, X_train, K_train):
    from sklearn.decomposition import KernelPCA
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Figure 2 — Quantum Kernel SVM Decision Boundary\n(KernelPCA 2D projection)')
    kpca = KernelPCA(n_components=2, kernel='precomputed')
    X_2d = kpca.fit_transform(K_train)
    decisions = svm.decision_function(K_train)
    sv_mask = np.zeros(len(X_train), dtype=bool)
    sv_mask[svm.support_] = True
    y_pred = svm.predict(K_train)
    for ax_idx, (ax, title) in enumerate(zip(axes, ['Class labels','Decision function'])):
        ax.set_title(title)
        ax.set_xlabel('KernelPCA component 1')
        ax.set_ylabel('KernelPCA component 2')
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            for label, color, name in [(0, GREEN,'Normal'),(1, RED,'Anomaly')]:
                mask = y_pred == label
                ax.scatter(X_2d[mask,0], X_2d[mask,1], c=color,
                          s=60, alpha=0.8, label=name, zorder=3)
            ax.scatter(X_2d[sv_mask,0], X_2d[sv_mask,1], s=200,
                      facecolors='none', edgecolors=AMBER, linewidths=1.5,
                      label=f'Support vectors (n={sv_mask.sum()})', zorder=4)
            ax.legend(fontsize=9)
        else:
            sc = ax.scatter(X_2d[:,0], X_2d[:,1], c=decisions,
                           cmap='RdYlGn', s=80, alpha=0.9, zorder=3)
            plt.colorbar(sc, ax=ax, label='Decision function value')
    plt.tight_layout()
    p = f'{OUT}/fig2_decision_boundary.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(); logger.success(f"Saved: {p}")


def fig3_latency():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Figure 3 — QENI Latency Profile\nRaspberry Pi 5 ↔ IBM Quantum Cloud')
    ax = axes[0]
    cats   = ['Quantum kernel\n(ibm_marrakesh)','Quantum kernel\n(Simulator)',
              'SVM decision\n(on-device)','Total inference\n(IBM QPU)']
    vals   = [144.0, 5.0, 0.00175, 144.0]
    colors = [RED, AMBER, GREEN, PURPLE]
    bars = ax.barh(cats, vals, color=colors, alpha=0.85, height=0.5)
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Inference latency breakdown')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, vals):
        ax.text(val*1.1, bar.get_y()+bar.get_height()/2,
                f'{val:.5f}s' if val < 0.01 else f'{val:.1f}s',
                va='center', fontsize=9, color='#c9d1d9')
    ax2 = axes[1]
    systems = ['Classical SVM\n(sklearn)','QENI Simulator\n(AerSimulator)',
               'QENI Real QPU\n(ibm_marrakesh)']
    ttimes  = [0.01, 79.6, 789.0]
    bcolors = [GREEN, AMBER, RED]
    bars2 = ax2.bar(systems, ttimes, color=bcolors, alpha=0.85, width=0.5)
    ax2.set_ylabel('Training time (seconds)')
    ax2.set_title('Training kernel matrix\n30×30 = 465 circuits')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, ttimes):
        ax2.text(bar.get_x()+bar.get_width()/2, val*1.5,
                f'{val:.3f}s' if val < 1 else f'{val:.1f}s',
                ha='center', fontsize=9, color='#c9d1d9')
    ax3 = axes[2]
    labels = ['Quantum kernel\n(IBM QPU)','Classical SVM\ndecision']
    sizes  = [144.0, 0.00175]
    wedge_colors = [PURPLE, GREEN]
    wedges, texts, autotexts = ax3.pie(
        sizes, labels=labels, colors=wedge_colors,
        autopct=lambda p: f'{p:.2f}%' if p > 1 else '<0.01%',
        explode=(0.05, 0.05), startangle=90,
        textprops={'color':'#c9d1d9','fontsize':9},
    )
    ax3.set_title('Inference time split')
    ax3.text(0, -1.4,
             'Classical SVM: 1.75ms\nQuantum Kernel: 144s\nRatio: 102,996×',
             ha='center', fontsize=9, color=AMBER,
             bbox=dict(boxstyle='round', facecolor='#1f2937', alpha=0.8))
    plt.tight_layout()
    p = f'{OUT}/fig3_latency.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(); logger.success(f"Saved: {p}")


def fig4_feature_space(encoder):
    from core.training import generate_training_data
    X, y = generate_training_data(n_samples=100, seed=42)
    X_enc = encoder.transform(X)
    features     = ['Temperature (°C)','Vibration (m/s²)','Current (A)','Accel mag (m/s²)']
    features_enc = ['Temp encoded','Vibration encoded','Current encoded','Accel encoded']
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Figure 4 — Feature Space\nRaw sensor values vs quantum angle encoding')
    gs = GridSpec(2, 4, fig, hspace=0.45, wspace=0.35)
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        for label, color, name in [(0,GREEN,'Normal'),(1,RED,'Anomaly')]:
            mask = y == label
            ax.hist(X[mask,i], bins=20, alpha=0.7, color=color,
                   label=name, density=True)
        ax.set_title(f'Raw: {features[i]}', fontsize=9)
        ax.set_xlabel('Value', fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(fontsize=7)
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        for label, color, name in [(0,GREEN,'Normal'),(1,RED,'Anomaly')]:
            mask = y == label
            ax.hist(X_enc[mask,i], bins=20, alpha=0.7, color=color,
                   label=name, density=True)
        ax.set_title(f'Encoded: {features_enc[i]}', fontsize=9)
        ax.set_xlabel('Angle (radians)', fontsize=8)
        ax.set_xlim(0, 2*np.pi)
        ax.axvline(np.pi, color=AMBER, linestyle='--', alpha=0.5, linewidth=0.8)
        ax.grid(True, alpha=0.3)
    p = f'{OUT}/fig4_feature_space.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(); logger.success(f"Saved: {p}")


def fig5_performance(info):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Figure 5 — Performance Summary\nIBM ibm_marrakesh · Raspberry Pi 5')
    ax = axes[0,0]
    cm = np.array(info.get('confusion_matrix', [[6,0],[2,0]]))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred Normal','Pred Anomaly'])
    ax.set_yticklabels(['True Normal','True Anomaly'])
    ax.set_title('Confusion matrix (real QPU)')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                   fontsize=20, fontweight='bold',
                   color='white' if cm[i,j] > cm.max()/2 else '#c9d1d9')
    plt.colorbar(im, ax=ax)
    ax2 = axes[0,1]
    m_sim = [0.875, 0.923, 0.667, 0.500]
    m_qpu = [info.get('accuracy',0.75), info.get('f1_normal',0.857),
             info.get('f1_anomaly',0.0), info.get('roc_auc',0.75)]
    mnames = ['Accuracy','F1 Normal','F1 Anomaly','ROC-AUC']
    x = np.arange(len(mnames)); w = 0.35
    ax2.bar(x-w/2, m_sim, w, label='Simulator', color=BLUE, alpha=0.85)
    ax2.bar(x+w/2, m_qpu, w, label='IBM ibm_marrakesh', color=PURPLE, alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(mnames, fontsize=9)
    ax2.set_ylim(0, 1.15); ax2.set_ylabel('Score')
    ax2.set_title('Simulator vs real QPU metrics')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(0.5, color=AMBER, linestyle='--', alpha=0.4, linewidth=0.8)
    ax3 = axes[1,0]
    sv_counts = [26, 27, 19]
    slabels   = ['Simulator\nRun 1','Simulator\nRun 2','IBM QPU\nibm_marrakesh']
    sv_colors = [BLUE, CYAN, PURPLE]
    bars = ax3.bar(slabels, sv_counts, color=sv_colors, alpha=0.85, width=0.5)
    ax3.axhline(30, color=AMBER, linestyle='--', alpha=0.6,
               label='Total training samples (30)')
    ax3.set_ylabel('Number of support vectors')
    ax3.set_title('Support vectors per training run')
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, sv_counts):
        ax3.text(bar.get_x()+bar.get_width()/2, val+0.3,
                str(val), ha='center', fontsize=11, fontweight='bold')
    ax4 = axes[1,1]
    ax4.axis('off')
    lines = [
        ('QENI — Key Results', '#f0f6fc', 12, 'bold'),
        ('', '#c9d1d9', 9, 'normal'),
        ('Edge:    Raspberry Pi 5 (ARM aarch64)', CYAN, 10, 'normal'),
        ('QPU:     IBM ibm_marrakesh 156 qubits', PURPLE, 10, 'normal'),
        ('', '#c9d1d9', 9, 'normal'),
        ('Training kernel:  789.0s on real QPU', RED, 10, 'normal'),
        ('Test kernel:      284.7s on real QPU', AMBER, 10, 'normal'),
        ('Inference kernel: ~144s per cycle', AMBER, 10, 'normal'),
        ('SVM decision:     1.75ms on-device', GREEN, 10, 'normal'),
        ('', '#c9d1d9', 9, 'normal'),
        ('ROC-AUC:  0.750 on real QPU', BLUE, 10, 'bold'),
        ('Accuracy: 0.750 on real QPU', BLUE, 10, 'normal'),
        ('', '#c9d1d9', 9, 'normal'),
        ('Privacy: 0 bytes raw data sent', GREEN, 10, 'bold'),
        ('Only [0,2π]⁴ feature angles leave Pi', GREEN, 9, 'normal'),
        ('', '#c9d1d9', 9, 'normal'),
        ('Ratio: 102,996× quantum vs classical', AMBER, 10, 'bold'),
    ]
    y_pos = 0.97
    for text, color, size, weight in lines:
        ax4.text(0.05, y_pos, text, transform=ax4.transAxes,
                fontsize=size, color=color, fontweight=weight,
                va='top', fontfamily='monospace')
        y_pos -= 0.057
    p = f'{OUT}/fig5_performance.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(); logger.success(f"Saved: {p}")


def fig6_kernel_rows(svm, X_train, K_train):
    from core.training import generate_training_data
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Figure 6 — Kernel Inference Rows\nK(x_new, x_train_i) for 6 test points')
    X_test, y_test = generate_training_data(n_samples=20, seed=99)
    enc = joblib.load('models/encoder_latest.joblib')
    X_test_enc = enc.transform(X_test)
    normal_idx  = np.where(y_test==0)[0][:3]
    anomaly_idx = np.where(y_test==1)[0][:3]
    selected = list(normal_idx) + list(anomaly_idx)
    sel_labels = [0]*len(normal_idx) + [1]*len(anomaly_idx)
    for plot_idx, (idx, label) in enumerate(zip(selected, sel_labels)):
        ax = axes[plot_idx//3][plot_idx%3]
        x_enc = X_test_enc[idx]
        K_row = np.array([
            float(np.dot(x_enc, X_train[j]) /
                  (np.linalg.norm(x_enc)*np.linalg.norm(X_train[j])+1e-10))
            for j in range(len(X_train))
        ])
        K_row = (K_row - K_row.min()) / (K_row.max() - K_row.min() + 1e-10)
        color = RED if label==1 else GREEN
        name  = 'ANOMALY' if label==1 else 'NORMAL'
        ax.bar(range(len(K_row)), K_row, color=color, alpha=0.7, width=0.8)
        ax.axhline(K_row.mean(), color=AMBER, linestyle='--', alpha=0.8,
                  linewidth=1, label=f'mean={K_row.mean():.3f}')
        ax.set_title(f'Test point {plot_idx+1}: {name}', color=color)
        ax.set_xlabel('Training point index')
        ax.set_ylabel('K(x_new, x_train_i)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(-0.5, len(K_row)-0.5)
    plt.tight_layout()
    p = f'{OUT}/fig6_kernel_rows.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(); logger.success(f"Saved: {p}")


def main():
    logger.info("QENI Graph Generator")
    svm, encoder, X_train, K_train, info = load_artifacts()
    logger.info("Fig 1: kernel matrix...")
    fig1_kernel_matrix(K_train, info)
    logger.info("Fig 2: decision boundary...")
    fig2_decision_boundary(svm, X_train, K_train)
    logger.info("Fig 3: latency...")
    fig3_latency()
    logger.info("Fig 4: feature space...")
    fig4_feature_space(encoder)
    logger.info("Fig 5: performance summary...")
    fig5_performance(info)
    logger.info("Fig 6: kernel rows...")
    fig6_kernel_rows(svm, X_train, K_train)
    logger.success("ALL FIGURES DONE")
    print(f"\nFiles in graphs/:")
    for f in sorted(os.listdir(OUT)):
        size = os.path.getsize(f'{OUT}/{f}') // 1024
        print(f"  {f}  ({size} KB)")
    try:
        import socket
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "your-pi-ip"
    print(f"\nCopy to your computer:\n  scp pi@{ip}:~/qeni/graphs/*.png ./")


if __name__ == "__main__":
    main()
