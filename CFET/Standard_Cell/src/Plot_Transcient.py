import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
FIGURE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "figure"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# A (high-quality) random transistor from the dataset
PARAMS = {
    'Vth_N': 0.244, 'Ion_N': 3.11e-05, 'Ioff_N': 3.32e-11, 'SS_N': 68.63,
    'Vth_P': -0.255, 'Ion_P': 1.98e-05, 'Ioff_P': 1.62e-11, 'SS_P': 68.81
}

CLOADS = [0.5, 1.0, 2.0, 4.0]

LOAD_COLORS = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'] 

def generate_transient_netlist(gate_type, cload):
    netlist = f"""* CFET Transient Viewer - {gate_type}
.control
    pre_osdi CFET_comportemental.osdi
.endc

v_dd vdd 0 dc 0.7
v_in_a in_a 0 PULSE(0 0.7 20p 2p 2p 150p 300p)
"""
    if gate_type == "INV":
        netlist += """
v_in_b in_b 0 dc 0
N_pmos out in_a vdd vdd pmos_cfet
N_nmos out in_a 0 0 nmos_cfet
"""
    elif gate_type == "NAND2":
        netlist += """
v_in_b in_b 0 dc 0.7
N_pmos1 out in_a vdd vdd pmos_cfet
N_pmos2 out in_b vdd vdd pmos_cfet
N_nmos1 out in_a n_mid 0 nmos_cfet
N_nmos2 n_mid in_b 0 0 nmos_cfet
"""
    elif gate_type == "NOR2":
        netlist += """
v_in_b in_b 0 dc 0
N_pmos1 p_mid in_a vdd vdd pmos_cfet
N_pmos2 out in_b p_mid vdd pmos_cfet
N_nmos1 out in_a 0 0 nmos_cfet
N_nmos2 out in_b 0 0 nmos_cfet
"""

    netlist += f"""
c_load out 0 {cload}f

.model pmos_cfet cfet_device type=-1 vth={abs(PARAMS['Vth_P'])} ion={PARAMS['Ion_P']} ioff={PARAMS['Ioff_P']} ss={PARAMS['SS_P']}
.model nmos_cfet cfet_device type=1 vth={PARAMS['Vth_N']} ion={PARAMS['Ion_N']} ioff={PARAMS['Ioff_N']} ss={PARAMS['SS_N']}

.control
    tran 0.1p 200p
    wrdata {gate_type}_wave.txt v(in_a) v(in_b) v(out)
    quit
.endc
.end
"""
    return netlist

def get_crossing_time(t, v, thresh=0.35, direction='rise'):
    """Trouve le moment exact où le signal croise 50% de VDD par interpolation"""
    for i in range(len(v)-1):
        if direction == 'rise' and v[i] <= thresh and v[i+1] > thresh:
            slope = (v[i+1] - v[i]) / (t[i+1] - t[i])
            return t[i] + (thresh - v[i]) / slope
        elif direction == 'fall' and v[i] >= thresh and v[i+1] < thresh:
            slope = (v[i+1] - v[i]) / (t[i+1] - t[i])
            return t[i] + (thresh - v[i]) / slope
    return None

def plot_waveforms():
    gates = ["INV", "NAND2", "NOR2"]

    for gate in gates:
        plt.figure(figsize=(11, 7))
        sns.set_theme(style="whitegrid", context="talk")
        
        input_plotted = False
        t_in_ref = None

        for idx, cload in enumerate(CLOADS):
            color = LOAD_COLORS[idx]
            sp_path = os.path.join(DATA_DIR, f"{gate}_temp.sp")
            wave_path = os.path.join(DATA_DIR, f"{gate}_wave.txt")
            
            with open(sp_path, 'w') as f:
                f.write(generate_transient_netlist(gate, cload))
                
            subprocess.run(["ngspice", "-b", sp_path], cwd=DATA_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(wave_path):
                data = pd.read_csv(wave_path, sep='\s+', header=None)
                time_ps = data[0].values * 1e12
                v_in_a = data[1].values
                v_out = data[5].values

                if not input_plotted:
                    plt.plot(time_ps, v_in_a, 'k--', linewidth=2, label="Input A (Toggle)", alpha=0.8)
                    t_in_ref = get_crossing_time(time_ps, v_in_a, 0.35, 'rise')
                    if t_in_ref:
                        plt.axvline(t_in_ref, color='k', linestyle=':', alpha=0.5)
                    input_plotted = True

                plt.plot(time_ps, v_out, color=color, linewidth=2.5, label=f"Output ({cload} fF)")

                t_out = get_crossing_time(time_ps, v_out, 0.35, 'fall')
                
                if t_in_ref and t_out:
                    delay = t_out - t_in_ref

                    y_arrow = 0.42 + (idx * 0.07)

                    plt.plot([t_out, t_out], [0.35, y_arrow], color=color, linestyle=':', alpha=0.7)

                    plt.annotate('', xy=(t_in_ref, y_arrow), xytext=(t_out, y_arrow),
                                 arrowprops=dict(arrowstyle='<->', color=color, lw=2))

                    plt.text((t_in_ref + t_out)/2, y_arrow + 0.01, f"{delay:.2f} ps", 
                             ha='center', va='bottom', fontsize=11, color=color, weight='bold')
                
                os.remove(sp_path)
                os.remove(wave_path)

        plt.axhline(0.35, color='gray', linestyle=':', alpha=0.7, label="50% VDD")

        plt.title(f"{gate} Transient Response - Multi-Load Analysis")
        plt.xlabel("Time (ps)")
        plt.ylabel("Voltage (V)")
        plt.xlim(15, 95)
        plt.ylim(-0.05, 0.75)
        plt.legend(loc='lower left', fontsize=11, framealpha=0.9)
        plt.tight_layout()

        save_path = os.path.join(FIGURE_DIR, f"Transient_{gate}_MultiLoad.png")
        plt.savefig(save_path, dpi=300)
        print(f"Figure {gate} Multi-Load générée : {save_path}")
        
        plt.show()

if __name__ == "__main__":
    print("Running multi-load transient simulations...")
    plot_waveforms()