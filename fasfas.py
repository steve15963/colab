import torch
import time

# GPU 워밍업
def warmup(model, inputs, iterations=10):
    for _ in range(iterations):
        model(inputs)
    torch.cuda.synchronize()

# 모델 벤치마크
def benchmark_model(model, inputs, iterations=1000):
    warmup(model, inputs)
    
    start_time = time.time()
    for _ in range(iterations):
        model(inputs)
    torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / iterations

# ResNet-50 예시
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).cuda()
model.eval()

# 다양한 배치 크기로 테스트
batch_sizes = [128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,]
for batch_size in batch_sizes:
    dummy_input = torch.randn(batch_size, 3, 224, 224).cuda()
    time_per_iter = benchmark_model(model, dummy_input)
    print(f"배치 크기 {batch_size}: {time_per_iter * 1000:.2f} ms/iter, {batch_size/time_per_iter:.2f} img/sec")
