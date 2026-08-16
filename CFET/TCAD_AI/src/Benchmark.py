import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import torch.nn as nn
import onnxruntime as ort
import warnings

warnings.filterwarnings("ignore")

class CFET_AI(nn.Module):
    def __init__(self, input_dim=5, output_dim=8, hidden_size=16):
        super().__init__()
        
        self.act = nn.ELU() 
        
        self.in_l = nn.Linear(input_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.out_l = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        x = self.act(self.in_l(x))
        x = self.act(self.fc2(self.act(self.fc1(x))) + x)
        x = self.act(self.fc4(self.act(self.fc3(x))) + x)
        return self.out_l(x)

def measure_inference(func, duration=2.0, warmup=100):
    for _ in range(warmup):
        func()
    
    iterations = 0
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < duration:
        func()
        iterations += 1
    end_time = time.perf_counter()
    
    return ((end_time - start_time) / iterations) * 1000.0

def run_benchmark(model, batch_size):
    print(f"\n[INFO] Running benchmark with {batch_size} component(s)...")
    
    model.to('cpu')
    
    dummy_input = torch.randn(batch_size, 5)
    onnx_path = f"temp_batch_{batch_size}.onnx"

    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, verbose=False,
        input_names=['input'], output_names=['output']
    )

    input_cpu = dummy_input.clone()
    def run_pt_cpu():
        with torch.no_grad():
            model(input_cpu)
    
    time_cpu_total = measure_inference(run_pt_cpu)

    time_cuda_total = None
    if torch.cuda.is_available():
        model.to('cuda')
        input_cuda = dummy_input.to('cuda')
        def run_pt_cuda():
            with torch.no_grad():
                model(input_cuda)
                torch.cuda.synchronize()
        time_cuda_total = measure_inference(run_pt_cuda)

    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_input = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    def run_onnx():
        ort_session.run(None, ort_input)
        
    time_onnx_total = measure_inference(run_onnx)

    if os.path.exists(onnx_path):
        os.remove(onnx_path)

    t_onnx_comp = time_onnx_total / batch_size
    t_cpu_comp = time_cpu_total / batch_size
    
    print("==================================================")
    print(f"      PERFORMANCE RESULTS (Batch = {batch_size})      ")
    print("==================================================")
    print(f"ONNX (CPU):      {t_onnx_comp:.8f} ms / component")
    print(f"PyTorch (CPU):   {t_cpu_comp:.8f} ms / component")
    
    if time_cuda_total is not None:
        t_cuda_comp = time_cuda_total / batch_size
        print(f"PyTorch (CUDA):  {t_cuda_comp:.8f} ms / component")
    print("==================================================")


if __name__ == "__main__":
    my_model = CFET_AI()
    my_model.eval()
    
    run_benchmark(my_model, batch_size=1)
    run_benchmark(my_model, batch_size=100000)