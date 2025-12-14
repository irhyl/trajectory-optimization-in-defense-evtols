import os, shutil, pandas as pd, matplotlib.pyplot as plt

cwd = os.getcwd()
src = os.path.join(cwd, 'notebooks', 'outputs', 'section7_hover.csv')
repo_out = os.path.join(cwd, 'outputs')
os.makedirs(repo_out, exist_ok=True)
try:
    dst = os.path.join(repo_out, 'section7_hover.csv')
    shutil.copy(src, dst)
    df = pd.read_csv(src)
    # altitude plot
    p1 = os.path.join(repo_out, 'section7_hover_altitude.png')
    plt.figure(figsize=(6,3))
    plt.plot(df['time'], df['z'], label='z (m)')
    plt.xlabel('time (s)'); plt.ylabel('altitude (m)'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(p1); plt.close()
    # power plot
    p2 = os.path.join(repo_out, 'section7_hover_power.png')
    plt.figure(figsize=(6,3))
    plt.plot(df['time'], df['P_elec'], label='P_elec (W)', color='tab:orange')
    plt.xlabel('time (s)'); plt.ylabel('power (W)'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(p2); plt.close()
    print('Copied CSV to', dst)
    print('Saved plots:', p1, p2)
    print('\nCSV sample:')
    print(df.head().to_string(index=False))
except Exception as e:
    print('Failed to copy or plot hover CSV:', e)
