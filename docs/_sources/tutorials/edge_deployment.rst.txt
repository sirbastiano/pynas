Edge Deployment
===============

This tutorial covers deploying PyNAS-evolved neural networks on edge devices, including optimization techniques, deployment strategies, and performance monitoring.

Overview
--------

Deploying neural networks on edge devices requires careful consideration of:

- **Resource Constraints**: Limited memory, compute, and power
- **Optimization**: Model compression and acceleration techniques
- **Deployment**: Framework selection and integration
- **Monitoring**: Performance tracking and maintenance

Edge Device Considerations
--------------------------

Hardware Constraints
~~~~~~~~~~~~~~~~~~~~

Understanding target device limitations:

.. code-block:: python

    import psutil
    import torch
    
    class EdgeDeviceProfiler:
        """
        Profile edge device capabilities.
        """
        
        def __init__(self):
            self.device_info = self.get_device_info()
        
        def get_device_info(self):
            """Collect device information."""
            info = {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'has_gpu': torch.cuda.is_available(),
                'gpu_memory': self.get_gpu_memory() if torch.cuda.is_available() else 0
            }
            return info
        
        def get_gpu_memory(self):
            """Get GPU memory information."""
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory
            return 0
        
        def estimate_model_requirements(self, model):
            """Estimate model resource requirements."""
            # Calculate model size
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size = param_size + buffer_size
            
            # Estimate inference memory (rough approximation)
            inference_memory = model_size * 2  # Parameters + activations
            
            return {
                'model_size_mb': model_size / 1024 / 1024,
                'estimated_inference_memory_mb': inference_memory / 1024 / 1024,
                'parameter_count': sum(p.numel() for p in model.parameters())
            }
        
        def check_compatibility(self, model):
            """Check if model can run on device."""
            requirements = self.estimate_model_requirements(model)
            available_memory = self.device_info['memory_available'] / 1024 / 1024
            
            return {
                'can_run': requirements['estimated_inference_memory_mb'] < available_memory * 0.8,
                'memory_utilization': requirements['estimated_inference_memory_mb'] / available_memory,
                'recommendations': self.get_optimization_recommendations(requirements)
            }
        
        def get_optimization_recommendations(self, requirements):
            """Provide optimization recommendations."""
            recommendations = []
            
            if requirements['model_size_mb'] > 50:
                recommendations.append('Consider model quantization')
            if requirements['parameter_count'] > 1e6:
                recommendations.append('Apply pruning to reduce parameters')
            if requirements['estimated_inference_memory_mb'] > 100:
                recommendations.append('Use gradient checkpointing')
            
            return recommendations

Model Optimization
------------------

Quantization
~~~~~~~~~~~~

Reduce model precision for faster inference:

.. code-block:: python

    import torch.quantization as quant
    
    class ModelQuantizer:
        """
        Quantize models for edge deployment.
        """
        
        def __init__(self, model):
            self.model = model
        
        def prepare_for_quantization(self):
            """Prepare model for quantization."""
            # Set model to evaluation mode
            self.model.eval()
            
            # Specify quantization configuration
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Prepare model
            model_prepared = torch.quantization.prepare(self.model)
            return model_prepared
        
        def calibrate_model(self, model_prepared, calibration_loader):
            """Calibrate model with representative data."""
            model_prepared.eval()
            
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(calibration_loader):
                    if batch_idx >= 100:  # Use 100 batches for calibration
                        break
                    model_prepared(data)
            
            return model_prepared
        
        def quantize_model(self, model_prepared):
            """Convert to quantized model."""
            model_quantized = torch.quantization.convert(model_prepared)
            return model_quantized
        
        def compare_models(self, original_model, quantized_model, test_loader):
            """Compare original and quantized model performance."""
            original_size = self.get_model_size(original_model)
            quantized_size = self.get_model_size(quantized_model)
            
            original_accuracy = self.evaluate_model(original_model, test_loader)
            quantized_accuracy = self.evaluate_model(quantized_model, test_loader)
            
            return {
                'size_reduction': (original_size - quantized_size) / original_size,
                'accuracy_drop': original_accuracy - quantized_accuracy,
                'original_size_mb': original_size / 1024 / 1024,
                'quantized_size_mb': quantized_size / 1024 / 1024
            }
        
        def get_model_size(self, model):
            """Calculate model size in bytes."""
            return sum(p.numel() * p.element_size() for p in model.parameters())
        
        def evaluate_model(self, model, test_loader):
            """Evaluate model accuracy."""
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, targets in test_loader:
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            return correct / total

Pruning
~~~~~~~

Remove redundant parameters:

.. code-block:: python

    import torch.nn.utils.prune as prune
    
    class ModelPruner:
        """
        Prune models for edge deployment.
        """
        
        def __init__(self, model):
            self.model = model
        
        def structured_pruning(self, pruning_ratio=0.3):
            """Apply structured pruning to remove entire channels."""
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.ln_structured(
                        module, name='weight', amount=pruning_ratio, 
                        n=2, dim=0  # Prune output channels
                    )
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
        
        def unstructured_pruning(self, pruning_ratio=0.5):
            """Apply unstructured pruning to remove individual weights."""
            parameters_to_prune = []
            
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    parameters_to_prune.append((module, 'weight'))
            
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
        
        def gradual_pruning(self, initial_sparsity=0.0, final_sparsity=0.8, 
                          num_iterations=10):
            """Apply gradual pruning over multiple iterations."""
            current_sparsity = initial_sparsity
            sparsity_increment = (final_sparsity - initial_sparsity) / num_iterations
            
            for iteration in range(num_iterations):
                # Apply pruning
                self.unstructured_pruning(current_sparsity)
                
                # Fine-tune model (implement your training loop here)
                # self.fine_tune_model()
                
                current_sparsity += sparsity_increment
                print(f"Iteration {iteration + 1}: Sparsity = {current_sparsity:.2f}")
        
        def remove_pruning_masks(self):
            """Permanently remove pruned weights."""
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
        
        def analyze_sparsity(self):
            """Analyze current model sparsity."""
            total_params = 0
            zero_params = 0
            
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight'):
                    total_params += module.weight.numel()
                    zero_params += (module.weight == 0).sum().item()
            
            sparsity = zero_params / total_params if total_params > 0 else 0
            return {
                'total_parameters': total_params,
                'zero_parameters': zero_params,
                'sparsity_ratio': sparsity,
                'compression_ratio': 1 / (1 - sparsity) if sparsity < 1 else float('inf')
            }

Knowledge Distillation
~~~~~~~~~~~~~~~~~~~~~~

Transfer knowledge from larger models:

.. code-block:: python

    import torch.nn.functional as F
    
    class KnowledgeDistiller:
        """
        Distill knowledge from teacher to student model.
        """
        
        def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7):
            self.teacher = teacher_model
            self.student = student_model
            self.temperature = temperature
            self.alpha = alpha  # Weight for distillation loss
            
            # Freeze teacher model
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
        
        def distillation_loss(self, student_outputs, teacher_outputs, targets):
            """Calculate combined distillation and task loss."""
            # Soft targets from teacher
            soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
            soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
            
            # Distillation loss
            distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
            distill_loss *= (self.temperature ** 2)
            
            # Task loss
            task_loss = F.cross_entropy(student_outputs, targets)
            
            # Combined loss
            total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss
            
            return total_loss, distill_loss, task_loss
        
        def train_student(self, train_loader, optimizer, num_epochs=10):
            """Train student model with distillation."""
            self.student.train()
            
            for epoch in range(num_epochs):
                total_loss = 0
                for batch_idx, (data, targets) in enumerate(train_loader):
                    optimizer.zero_grad()
                    
                    # Get predictions
                    with torch.no_grad():
                        teacher_outputs = self.teacher(data)
                    student_outputs = self.student(data)
                    
                    # Calculate loss
                    loss, distill_loss, task_loss = self.distillation_loss(
                        student_outputs, teacher_outputs, targets
                    )
                    
                    # Backpropagation
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    if batch_idx % 100 == 0:
                        print(f'Epoch {epoch}, Batch {batch_idx}: '
                              f'Total Loss: {loss.item():.4f}, '
                              f'Distill Loss: {distill_loss.item():.4f}, '
                              f'Task Loss: {task_loss.item():.4f}')
                
                avg_loss = total_loss / len(train_loader)
                print(f'Epoch {epoch}: Average Loss = {avg_loss:.4f}')

Deployment Frameworks
---------------------

TensorRT Optimization
~~~~~~~~~~~~~~~~~~~~~

Optimize for NVIDIA devices:

.. code-block:: python

    import tensorrt as trt
    import torch
    from torch2trt import torch2trt
    
    class TensorRTDeployer:
        """
        Deploy models using TensorRT optimization.
        """
        
        def __init__(self, model, input_shape=(1, 3, 224, 224)):
            self.model = model
            self.input_shape = input_shape
        
        def convert_to_tensorrt(self, fp16_mode=True, max_batch_size=1):
            """Convert PyTorch model to TensorRT."""
            # Create dummy input
            dummy_input = torch.randn(self.input_shape).cuda()
            
            # Convert model
            model_trt = torch2trt(
                self.model.cuda(),
                [dummy_input],
                fp16_mode=fp16_mode,
                max_batch_size=max_batch_size
            )
            
            return model_trt
        
        def benchmark_models(self, original_model, trt_model, num_runs=100):
            """Benchmark original vs TensorRT model."""
            dummy_input = torch.randn(self.input_shape).cuda()
            
            # Benchmark original model
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            for _ in range(num_runs):
                _ = original_model(dummy_input)
            end_time.record()
            torch.cuda.synchronize()
            original_time = start_time.elapsed_time(end_time) / num_runs
            
            # Benchmark TensorRT model
            start_time.record()
            for _ in range(num_runs):
                _ = trt_model(dummy_input)
            end_time.record()
            torch.cuda.synchronize()
            trt_time = start_time.elapsed_time(end_time) / num_runs
            
            return {
                'original_time_ms': original_time,
                'tensorrt_time_ms': trt_time,
                'speedup': original_time / trt_time
            }

ONNX Deployment
~~~~~~~~~~~~~~~

Deploy using ONNX for cross-platform compatibility:

.. code-block:: python

    import torch
    import onnx
    import onnxruntime as ort
    
    class ONNXDeployer:
        """
        Deploy models using ONNX format.
        """
        
        def __init__(self, model, input_shape=(1, 3, 224, 224)):
            self.model = model
            self.input_shape = input_shape
        
        def export_to_onnx(self, output_path, opset_version=11):
            """Export PyTorch model to ONNX."""
            dummy_input = torch.randn(self.input_shape)
            
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            return output_path
        
        def optimize_onnx(self, onnx_path, optimized_path):
            """Optimize ONNX model for inference."""
            # Load and optimize
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create optimized session
            session = ort.InferenceSession(onnx_path, sess_options)
            
            # Save optimized model
            session.save(optimized_path)
            
            return optimized_path
        
        def benchmark_onnx(self, onnx_path, num_runs=100):
            """Benchmark ONNX model performance."""
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            
            dummy_input = torch.randn(self.input_shape).numpy()
            
            # Warmup
            for _ in range(10):
                _ = session.run(None, {input_name: dummy_input})
            
            # Benchmark
            import time
            start_time = time.time()
            for _ in range(num_runs):
                _ = session.run(None, {input_name: dummy_input})
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
            
            return {
                'average_inference_time_ms': avg_time,
                'throughput_fps': 1000 / avg_time
            }

Mobile Deployment
-----------------

PyTorch Mobile
~~~~~~~~~~~~~~

Deploy on mobile devices:

.. code-block:: python

    import torch
    from torch.utils.mobile_optimizer import optimize_for_mobile
    
    class MobileDeployer:
        """
        Deploy models for mobile devices.
        """
        
        def __init__(self, model):
            self.model = model
        
        def prepare_for_mobile(self, input_shape=(1, 3, 224, 224)):
            """Prepare model for mobile deployment."""
            # Set to evaluation mode
            self.model.eval()
            
            # Trace the model
            dummy_input = torch.randn(input_shape)
            traced_model = torch.jit.trace(self.model, dummy_input)
            
            # Optimize for mobile
            mobile_model = optimize_for_mobile(traced_model)
            
            return mobile_model
        
        def save_mobile_model(self, mobile_model, output_path):
            """Save mobile-optimized model."""
            mobile_model._save_for_lite_interpreter(output_path)
            return output_path
        
        def validate_mobile_model(self, mobile_path, test_loader):
            """Validate mobile model accuracy."""
            # Load mobile model
            mobile_model = torch.jit.load(mobile_path)
            mobile_model.eval()
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, targets in test_loader:
                    outputs = mobile_model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            accuracy = correct / total
            return accuracy

Performance Monitoring
----------------------

Real-time Performance Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monitor deployed model performance:

.. code-block:: python

    import time
    import psutil
    import threading
    from collections import deque
    
    class PerformanceMonitor:
        """
        Monitor deployed model performance.
        """
        
        def __init__(self, model, window_size=100):
            self.model = model
            self.window_size = window_size
            self.inference_times = deque(maxlen=window_size)
            self.memory_usage = deque(maxlen=window_size)
            self.cpu_usage = deque(maxlen=window_size)
            self.monitoring = False
            self.monitor_thread = None
        
        def wrapped_inference(self, input_data):
            """Wrapped inference with performance monitoring."""
            start_time = time.time()
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_data)
            
            # Record metrics
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            self.inference_times.append(inference_time)
            
            return output
        
        def start_monitoring(self, interval=1.0):
            """Start background performance monitoring."""
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_system, args=(interval,)
            )
            self.monitor_thread.start()
        
        def stop_monitoring(self):
            """Stop background monitoring."""
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()
        
        def _monitor_system(self, interval):
            """Background system monitoring."""
            while self.monitoring:
                # Record memory usage
                memory_info = psutil.virtual_memory()
                self.memory_usage.append(memory_info.percent)
                
                # Record CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu_percent)
                
                time.sleep(interval)
        
        def get_performance_stats(self):
            """Get current performance statistics."""
            stats = {}
            
            if self.inference_times:
                stats['inference'] = {
                    'mean_time_ms': sum(self.inference_times) / len(self.inference_times),
                    'min_time_ms': min(self.inference_times),
                    'max_time_ms': max(self.inference_times),
                    'throughput_fps': 1000 / (sum(self.inference_times) / len(self.inference_times))
                }
            
            if self.memory_usage:
                stats['memory'] = {
                    'mean_usage_percent': sum(self.memory_usage) / len(self.memory_usage),
                    'peak_usage_percent': max(self.memory_usage)
                }
            
            if self.cpu_usage:
                stats['cpu'] = {
                    'mean_usage_percent': sum(self.cpu_usage) / len(self.cpu_usage),
                    'peak_usage_percent': max(self.cpu_usage)
                }
            
            return stats

Complete Deployment Pipeline
----------------------------

Here's a complete example of the deployment pipeline:

.. code-block:: python

    def deploy_nas_model(model_path, target_device='mobile'):
        """
        Complete deployment pipeline for NAS-evolved models.
        """
        
        # Load trained model
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        # Profile device capabilities
        profiler = EdgeDeviceProfiler()
        compatibility = profiler.check_compatibility(model)
        
        if not compatibility['can_run']:
            print("Model optimization required for target device")
            
            # Apply quantization
            quantizer = ModelQuantizer(model)
            model_prepared = quantizer.prepare_for_quantization()
            # Note: You would need calibration data here
            model = quantizer.quantize_model(model_prepared)
            
            # Apply pruning if still too large
            pruner = ModelPruner(model)
            pruner.unstructured_pruning(pruning_ratio=0.3)
            pruner.remove_pruning_masks()
        
        # Choose deployment strategy
        if target_device == 'mobile':
            deployer = MobileDeployer(model)
            mobile_model = deployer.prepare_for_mobile()
            output_path = 'model_mobile.ptl'
            deployer.save_mobile_model(mobile_model, output_path)
            
        elif target_device == 'nvidia':
            deployer = TensorRTDeployer(model)
            trt_model = deployer.convert_to_tensorrt()
            output_path = 'model_tensorrt.pth'
            torch.save(trt_model, output_path)
            
        else:  # ONNX for general deployment
            deployer = ONNXDeployer(model)
            output_path = 'model.onnx'
            deployer.export_to_onnx(output_path)
            optimized_path = 'model_optimized.onnx'
            deployer.optimize_onnx(output_path, optimized_path)
            output_path = optimized_path
        
        # Setup performance monitoring
        monitor = PerformanceMonitor(model)
        monitor.start_monitoring()
        
        print(f"Model deployed successfully: {output_path}")
        return output_path, monitor
    
    if __name__ == "__main__":
        model_path = "best_nas_model.pth"
        deployed_path, monitor = deploy_nas_model(model_path, target_device='mobile')
        
        # Monitor for a while
        time.sleep(60)
        stats = monitor.get_performance_stats()
        print(f"Performance stats: {stats}")
        monitor.stop_monitoring()

Best Practices
--------------

1. **Profile First**: Always profile target devices before deployment
2. **Gradual Optimization**: Apply optimizations incrementally
3. **Validate Accuracy**: Check model accuracy after each optimization
4. **Monitor Performance**: Continuously monitor deployed models
5. **A/B Testing**: Compare optimized vs original models in production

.. seealso::
   
   - :doc:`custom_architectures` for creating edge-optimized architectures
   - :doc:`../examples/edge_optimization` for edge optimization examples
   - :doc:`../api/train` for training API reference
