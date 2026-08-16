import os
import subprocess
import pandas as pd
import re
from tqdm import tqdm
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
INPUT_CSV = os.path.join(DATA_DIR, "Spice_input.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "Spice_output.csv")

VDD = 0.7
C_LOAD_FF = 1.0
GATE_TYPES = ["INV", "NAND2", "NOR2"]

GATE_SUBCKT = {
    "INV":   ("my_inv",   "{node} out vdd 0"),
    "NAND2": ("my_nand2", "{node} vdd out vdd 0"),
    "NOR2":  ("my_nor2",  "{node} 0 out vdd 0"),
}

SUBCKTS = """
.subckt my_inv in out vdd gnd
    N_pmos out in vdd vdd pmos_cfet
    N_nmos out in gnd gnd nmos_cfet
.ends

.subckt my_nand2 inA inB out vdd gnd
    N_pmos1 out inA vdd vdd pmos_cfet
    N_pmos2 out inB vdd vdd pmos_cfet
    N_nmos1 out inA n_mid gnd nmos_cfet
    N_nmos2 n_mid inB gnd gnd nmos_cfet
.ends

.subckt my_nor2 inA inB out vdd gnd
    N_pmos1 out inA p_mid vdd pmos_cfet
    N_pmos2 p_mid inB vdd vdd pmos_cfet
    N_nmos1 out inA gnd gnd nmos_cfet
    N_nmos2 out inB gnd gnd nmos_cfet
.ends
"""

def generate_gate_netlist(gate, params):
    subckt_name, pin_template = GATE_SUBCKT[gate]
    dut_pins = pin_template.format(node="in")

    netlist = f"""* CFET Gate Delay Test - {gate} (Fixed {C_LOAD_FF}fF Load)
.control
    pre_osdi CFET_comportemental.osdi
.endc

.model pmos_cfet cfet_device type=-1 vth={abs(params['Vth_P'])} ion={params['Ion_P']} ioff={params['Ioff_P']} ss={params['SS_P']}
.model nmos_cfet cfet_device type=1 vth={abs(params['Vth_N'])} ion={params['Ion_N']} ioff={params['Ioff_N']} ss={params['SS_N']}
{SUBCKTS}

v_dd vdd 0 dc {VDD}
v_in in 0 PULSE(0 {VDD} 10p 0.2p 0.2p 200p 400p)

X_dut {dut_pins} {subckt_name}
C_out out 0 {C_LOAD_FF}f

.control
    tran 0.02p 400p
    meas tran t_pHL TRIG v(in) VAL={VDD/2} RISE=1 TARG v(out) VAL={VDD/2} FALL=1
    meas tran t_pLH TRIG v(in) VAL={VDD/2} FALL=1 TARG v(out) VAL={VDD/2} RISE=1
    meas tran i_leak_avg AVG i(v_dd) FROM=0 TO=8p
    quit
.endc
.end
"""
    return netlist

def parse_spice_log(log_path):
    metrics = {'t_pHL': None, 't_pLH': None, 'i_leak_avg': None}
    if not os.path.exists(log_path):
        return metrics

    with open(log_path, "r") as f:
        content = f.read()

    for key in metrics.keys():
        match = re.search(rf"{key}\s*=\s*([+-]?\d+\.?\d*[eE]?[+-]?\d+)", content, re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))

    return metrics

def run_characterization():
    print(f"Loading input data from: {INPUT_CSV}")
    df_in = pd.read_csv(INPUT_CSV)

    total_sims = len(df_in) * len(GATE_TYPES)
    completed = set()
    file_exists = os.path.exists(OUTPUT_CSV)

    if file_exists:
        try:
            df_out_existing = pd.read_csv(OUTPUT_CSV)
            valid = df_out_existing.dropna(subset=['t_delay'])
            completed = set(zip(valid['ID'], valid['Gate_Type']))
            print(f"Found {len(completed)} valid completed simulations. Resuming...")
        except Exception:
            file_exists = False

    kwargs = {}
    if os.name == 'nt':
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        kwargs['startupinfo'] = si
        kwargs['creationflags'] = 0x08000000

    sp_path = os.path.join(DATA_DIR, "temp_sim_Gate.sp")
    log_path = os.path.join(DATA_DIR, "temp_sim_Gate.log")

    fieldnames = [
        'ID', 'Gate_Type', 'Lch', 'Wch', 'Tch', 'Tox1', 'Tox2',
        'Ion_N', 'Ioff_N', 'Vth_N', 'SS_N',
        'Ion_P', 'Ioff_P', 'Vth_P', 'SS_P',
        't_delay', 'i_leak'
    ]

    mode = 'a' if file_exists else 'w'
    n_ok, n_fail = 0, 0

    with open(OUTPUT_CSV, mode=mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        with tqdm(total=total_sims, desc="SPICE Gate Char", unit="sim", ncols=100) as pbar:
            for index, row in df_in.iterrows():
                for gate in GATE_TYPES:
                    if (index, gate) in completed:
                        pbar.update(1)
                        continue

                    with open(sp_path, 'w') as f:
                        f.write(generate_gate_netlist(gate, row))

                    subprocess.run(
                        ["ngspice", "-b", "temp_sim_Gate.sp", "-o", "temp_sim_Gate.log"],
                        cwd=DATA_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs
                    )

                    metrics = parse_spice_log(log_path)
                    
                    t_delay = None
                    if metrics['t_pHL'] is not None and metrics['t_pLH'] is not None:
                        t_delay = (metrics['t_pHL'] + metrics['t_pLH']) / 2.0
                    elif metrics['t_pHL'] is not None: 
                        t_delay = metrics['t_pHL']
                    elif metrics['t_pLH'] is not None: 
                        t_delay = metrics['t_pLH']

                    if t_delay is not None: 
                        n_ok += 1
                    else: 
                        n_fail += 1

                    i_leak = abs(metrics['i_leak_avg']) if metrics['i_leak_avg'] is not None else None

                    writer.writerow({
                        'ID': index, 'Gate_Type': gate,
                        'Lch': row['Lch'], 'Wch': row['Wch'], 'Tch': row['Tch'],
                        'Tox1': row['Tox1'], 'Tox2': row['Tox2'],
                        'Ion_N': row['Ion_N'], 'Ioff_N': row['Ioff_N'], 'Vth_N': row['Vth_N'], 'SS_N': row['SS_N'],
                        'Ion_P': row['Ion_P'], 'Ioff_P': row['Ioff_P'], 'Vth_P': row['Vth_P'], 'SS_P': row['SS_P'],
                        't_delay': t_delay, 'i_leak': i_leak
                    })
                    csvfile.flush()
                    pbar.update(1)

    print(f"\n--- DIAGNOSTICS ---\nSuccessful: {n_ok}\nFailed: {n_fail}")
    if os.path.exists(sp_path): os.remove(sp_path)
    if os.path.exists(log_path): os.remove(log_path)

if __name__ == "__main__":
    run_characterization()