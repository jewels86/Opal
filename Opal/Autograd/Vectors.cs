using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.IR.Values;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using Opal.Mathematics;

namespace Opal.Autograd;

public static partial class Operations
{
    #region Vector Tensor Helpers
    public static VectorTensorStorage NewCpuVectorStorage(double[] vector) =>
        new CpuStorage<double[]>(vector, [vector.Length], vector.Length);
    public static VectorTensorStorage NewGpuVectorStorage(double[] vector)
    {
        var buffer = Accelerator.Allocate1D<double>(vector.Length);
        buffer.CopyFromCPU(vector);
        return new GpuVectorStorage(buffer);
    }
    public static VectorTensorStorage NewDefaultVectorStorage(double[] vector) => 
        GpuAvailable ? NewGpuVectorStorage(vector) : NewCpuVectorStorage(vector);
    public static VectorTensor NewVector(VectorTensorStorage storage, List<object>? inputs, Action<Tensor<VectorTensorStorage>> backwards,
        VectorTensorStorage gradient) =>
        new(storage, inputs, backwards, gradient);
    public static VectorTensor NewVector(double[] vector, double[] gradient) => 
        NewVector(NewDefaultVectorStorage(vector), null, _ => { }, NewDefaultVectorStorage(gradient));
    #endregion
    #region Kernels
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>> VectorAddKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>> VectorSubtractKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>> VectorMultiplyKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>> ScalarVectorMultiplyKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, int> VectorConcatKernel { get; private set; }

    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>, int> VectorSliceKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>> VectorNegateKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, double> VectorFillKernel { get; private set; }
    #endregion
    #region Helpers
    public static bool UseGpu(params VectorTensorStorage[] storages) => storages.Any(s => s is GpuVectorStorage) && GpuAvailable;
    public static GpuVectorStorage ToGpuVector(VectorTensorStorage storage) => storage as GpuVectorStorage ?? (GpuVectorStorage)storage.ToGpu();
    public static MemoryBuffer1D<double, Stride1D.Dense> AllocateBuffer(long length) => Accelerator.Allocate1D<double>(length);
    
    public static VectorTensor BinaryOp(
        VectorTensor a,
        VectorTensor b,
        Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>> gpuKernel,
        Func<double[], double[], double[]> cpuFallback,
        Action<VectorTensor, VectorTensor, VectorTensor> gradientFn)
    {
        if (UseGpu(a.Value, b.Value))
        {
            var gpuA = ToGpuVector(a.Value);
            var gpuB = ToGpuVector(b.Value);
        
            var resultBuffer = AllocateBuffer(gpuA.GpuData.Length);
        
            Queue.Enqueue(() => gpuKernel((int)gpuA.GpuData.Length, gpuA.GpuData.View, gpuB.GpuData.View, resultBuffer.View));
        
            var resultStorage = new GpuVectorStorage(resultBuffer);
            var gradStorage = new GpuVectorStorage(AllocateBuffer(resultBuffer.Length));
        
            return NewVector(resultStorage, [a, b], 
                output => gradientFn(a, b, output), gradStorage);
        }
        var result = cpuFallback(a.Value.ToHost(), b.Value.ToHost());
        return NewVector(
            NewCpuVectorStorage(result), 
            [a, b], 
            output => gradientFn(a, b, (VectorTensor)output), 
            NewCpuVectorStorage(new double[result.Length]));
    }

    public static VectorTensor UnaryOp(
        VectorTensor vector,
        Action<Index1D, ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>> gpuKernel,
        Func<double[], double[]> cpuFallback,
        Action<VectorTensor, VectorTensor> gradientFn)
    {
        if (UseGpu(vector.Value))
        {
            var gpuVector = ToGpuVector(vector.Value);
            var resultBuffer = AllocateBuffer(gpuVector.GpuData.Length);
            
            Queue.Enqueue(() => gpuKernel((int)gpuVector.GpuData.Length, gpuVector.GpuData.View, resultBuffer.View));
            
            var resultStorage = new GpuVectorStorage(resultBuffer);
            var gradStorage = new GpuVectorStorage(AllocateBuffer(resultBuffer.Length));
            return NewVector(
                resultStorage, [vector], 
                output => gradientFn(vector, output), 
                gradStorage);
        }
        var result = cpuFallback(vector.Value.ToHost());
        return NewVector(
            NewCpuVectorStorage(result), 
            [vector], 
            output => gradientFn(vector, (VectorTensor)output), 
            NewCpuVectorStorage(Vectors.Zeros(result.Length)));
    }
    #endregion
    #region Storage Helpers
    public static void AccumulateGradient(VectorTensorStorage gradient, VectorTensorStorage incomingGrad)
    {
        if (UseGpu(gradient, incomingGrad))
        {
            var gpuGrad = ToGpuVector(gradient);
            var gpuIncoming = ToGpuVector(incomingGrad);

            Queue.Enqueue(() => VectorAddKernel(
                (int)gpuGrad.GpuData.Length,
                gpuGrad.GpuData.View,
                gpuIncoming.GpuData.View,
                gpuGrad.GpuData.View));
        }
        else
        {
            var gradData = gradient.ToHost();
            var incomingData = incomingGrad.ToHost();
            gradient.CopyFrom(Vectors.Add(gradData, incomingData));
        }
    }

    public static VectorTensorStorage MultiplyStorage(VectorTensorStorage a, VectorTensorStorage b)
    {
        if (UseGpu(a, b))
        {
            var gpuA = ToGpuVector(a);
            var gpuB = ToGpuVector(b);
            
            Queue.Enqueue(() => VectorMultiplyKernel(
                (int)gpuA.GpuData.Length,
                gpuA.GpuData.View,
                gpuB.GpuData.View,
                gpuA.GpuData.View));
            
            return new GpuVectorStorage(gpuA.GpuData);
        }
        var aData = a.ToHost();
        var bData = b.ToHost();
        return NewCpuVectorStorage(Vectors.Multiply(aData, bData));
    }
    
    public static VectorTensorStorage ScaleVectorStorage(VectorTensorStorage vector, ScalarTensorStorage scalar)
    {
        if (UseGpu(vector))
        {
            var gpuVec = ToGpuVector(vector);
            var gpuScalar = ToGpuScalar(scalar);
            var result = AllocateBuffer(gpuVec.GpuData.Length);
            Queue.Enqueue(() => ScalarVectorMultiplyKernel(
                (int)gpuVec.GpuData.Length,
                gpuVec.GpuData.View,
                gpuScalar.GpuData.View,
                result.View));
            return new GpuVectorStorage(result);
        }
        var vecData = vector.ToHost();
        var scalarData = scalar.ToHost();
        return NewCpuVectorStorage(Vectors.Multiply(vecData, scalarData));
    }

    public static VectorTensorStorage NegateStorage(VectorTensorStorage vector)
    {
        if (UseGpu(vector))
        {
            var gpuVector = ToGpuVector(vector);
            var result = AllocateBuffer(gpuVector.GpuData.Length);
            Queue.Enqueue(() => VectorNegateKernel(
                (int)gpuVector.GpuData.Length,
                gpuVector.GpuData.View,
                result.View));
            return new GpuVectorStorage(result);;
        }
        var vecData = vector.ToHost();
        return NewCpuVectorStorage(Vectors.Negate(vecData));
    }

    public static VectorTensorStorage AddStorage(VectorTensorStorage a, VectorTensorStorage b)
    {
        if (!UseGpu(a, b)) return NewCpuVectorStorage(Vectors.Add(a.ToHost(), b.ToHost()));
        var gpuA = ToGpuVector(a);
        var gpuB = ToGpuVector(b);
        var result = AllocateBuffer(gpuA.GpuData.Length);
            
        Queue.Enqueue(() => VectorAddKernel(
            (int)gpuA.GpuData.Length,
            gpuA.GpuData.View,
            gpuB.GpuData.View,
            result.View));
        return new GpuVectorStorage(result);
    }
    public static VectorTensorStorage SubtractStorage(VectorTensorStorage a, VectorTensorStorage b)
    {
        if (!UseGpu(a, b)) return NewCpuVectorStorage(Vectors.Subtract(a.ToHost(), b.ToHost()));
        var gpuA = ToGpuVector(a);
        var gpuB = ToGpuVector(b);
        var result = AllocateBuffer(gpuA.GpuData.Length);
            
        Queue.Enqueue(() => VectorSubtractKernel(
            (int)gpuA.GpuData.Length,
            gpuA.GpuData.View,
            gpuB.GpuData.View,
            result.View));
        return new GpuVectorStorage(result);
    }
    #endregion
    
    #region Operations
    public static VectorTensor Add(VectorTensor a, VectorTensor b) => 
        BinaryOp(
            a, b, 
            VectorAddKernel, Vectors.Add, 
            (_, _, output) =>
            {
                AccumulateGradient(a.Gradient, output.Gradient);
                AccumulateGradient(b.Gradient, output.Gradient);
            });
    public static VectorTensor Multiply(VectorTensor a, VectorTensor b) =>
        BinaryOp(
            a, b,
            VectorMultiplyKernel, Vectors.Multiply, (_, _, output) =>
            {
                AccumulateGradient(a.Gradient, MultiplyStorage(b.Value, output.Gradient));
                AccumulateGradient(b.Gradient, MultiplyStorage(a.Value, output.Gradient));
            });
    public static ScalarTensor Dot(VectorTensor a, VectorTensor b)
    {
        if (UseGpu(a.Value, b.Value))
        {
            var gpuA = ToGpuVector(a.Value);
            var gpuB = ToGpuVector(b.Value);
            
            var product = AllocateBuffer(gpuA.GpuData.Length);
            
            Queue.Enqueue(() => VectorMultiplyKernel(
                (int)gpuA.GpuData.Length,
                gpuA.GpuData.View,
                gpuB.GpuData.View,
                product.View));
        
            var result = AllocateScalar();
            Queue.Enqueue(() => Accelerator.Reduce<double, AddDouble>(
                Accelerator.DefaultStream,
                product.View,
                result.View));
        
            var resultStorage = new GpuScalarStorage(result);
            var gradStorage = new GpuScalarStorage(AllocateScalar());
        
            return new ScalarTensor(resultStorage, [a, b], Backward, gradStorage);
        
            void Backward(ScalarTensor output)
            {
                AccumulateGradient(a.Gradient, ScaleVectorStorage(b.Value, output.Gradient));
                AccumulateGradient(b.Gradient, ScaleVectorStorage(a.Value, output.Gradient));
            }
        }
        else
        {
            var result = Vectors.Dot(a.Value.ToHost(), b.Value.ToHost());
            return new ScalarTensor(
                NewCpuScalarStorage(result),
                [a, b],
                Backward,
                NewCpuScalarStorage(0.0));
        
            void Backward(ScalarTensor output)
            {
                var outGrad = output.Gradient.ToHost();
                AccumulateGradient(a.Gradient, NewCpuVectorStorage(Vectors.Multiply(b.Value.ToHost(), outGrad)));
                AccumulateGradient(b.Gradient, NewCpuVectorStorage(Vectors.Multiply(a.Value.ToHost(), outGrad)));
            }
        }
    }
    public static VectorTensor Concat(VectorTensor a, VectorTensor b)
    {
        if (UseGpu(a.Value, b.Value))
        {
            var gpuA = ToGpuVector(a.Value);
            var gpuB = ToGpuVector(b.Value);
            
            int aLength = (int)gpuA.GpuData.Length;
            int bLength = (int)gpuB.GpuData.Length;
            int totalLength = aLength + bLength;
            
            var result = AllocateBuffer(totalLength);
            
            Queue.Enqueue(() => GpuKernels.VectorConcatKernel(
                totalLength,
                gpuA.GpuData.View,
                gpuB.GpuData.View,
                result.View,
                aLength));
            
            var resultStorage = new GpuVectorStorage(result);
            var gradStorage = new GpuVectorStorage(AllocateBuffer(totalLength));
            
            return new VectorTensor(resultStorage, [a, b], Backward, gradStorage);
            
            void Backward(VectorTensor output)
            {
                if (output.Gradient is GpuVectorStorage gpuOutGrad && a.Gradient is GpuVectorStorage && b.Gradient is GpuVectorStorage)
                {
                    var tempA = AllocateBuffer(aLength);
                    Queue.Enqueue(() => VectorSliceKernel(aLength, gpuOutGrad.GpuData.View, tempA.View, 0));
                    AccumulateGradient(a.Gradient, new GpuVectorStorage(tempA));
        
                    var tempB = AllocateBuffer(bLength);
                    Queue.Enqueue(() => VectorSliceKernel(bLength, gpuOutGrad.GpuData.View, tempB.View, aLength));
                    AccumulateGradient(b.Gradient, new GpuVectorStorage(tempB));
                }
                else
                {
                    var outGrad = output.Gradient.ToHost();
                    var gradA = outGrad[..aLength];
                    var gradB = outGrad[aLength..];
        
                    AccumulateGradient(a.Gradient, NewCpuVectorStorage(gradA));
                    AccumulateGradient(b.Gradient, NewCpuVectorStorage(gradB));
                }
            }
        }
        else
        {
            var result = Vectors.Concat(a.Value.ToHost(), b.Value.ToHost());
            return new VectorTensor(
                NewCpuVectorStorage(result),
                [a, b],
                Backward,
                NewCpuVectorStorage(new double[result.Length]));
            
            void Backward(VectorTensor output)
            {
                var outGrad = output.Gradient.ToHost();
                var aLen = a.Value.Shape[0];
                var gradA = outGrad[..aLen];
                var gradB = outGrad[aLen..];
                
                a.Gradient.CopyFrom(Vectors.Add(a.Gradient.ToHost(), gradA));
                b.Gradient.CopyFrom(Vectors.Add(b.Gradient.ToHost(), gradB));
            }
        }
    }

    public static VectorTensor Multiply(VectorTensor a, ScalarTensor scalar)
    {
        if (!UseGpu(a.Value))
        {
            var value = NewCpuVectorStorage(Vectors.Multiply(a.Value.ToHost(), scalar.Value.ToHost()));
            return new VectorTensor(
                value, [a, scalar], 
                (output) => AccumulateGradient(a.Gradient, ScaleVectorStorage(output.Gradient, scalar.Gradient)),
                NewCpuVectorStorage(Vectors.Ones(value.TotalElements)));
        }
        var gpuAStorage = ToGpuVector(a.Value);
        var gpuScalarStorage = ToGpuScalar(scalar.Value);
        var result = AllocateBuffer(gpuAStorage.TotalElements);
        
        Queue.Enqueue(() => ScalarVectorMultiplyKernel(
            gpuAStorage.TotalElements,
            gpuAStorage.GpuData.View,
            gpuScalarStorage.GpuData.View,
            result.View));
        
        return new VectorTensor(
            new GpuVectorStorage(result), [a, scalar], 
            output => AccumulateGradient(a.Gradient, ScaleVectorStorage(output.Gradient, scalar.Gradient)),
            NewCpuVectorStorage(Vectors.Ones(gpuAStorage.TotalElements)));
    }

    public static VectorTensor Negate(VectorTensor a) => UnaryOp(
        a, VectorNegateKernel, Vectors.Negate, 
        (_, output) => AccumulateGradient(a.Gradient, NegateStorage(output.Gradient)));
    
    public static ScalarTensor Sum(VectorTensor vector)
    {
        if (UseGpu(vector.Value))
        {
            var gpuVec = ToGpuVector(vector.Value);
            var result = AllocateScalar();
            Queue.Enqueue(() => Accelerator.Reduce<double, AddDouble>(
                Accelerator.DefaultStream,
                gpuVec.GpuData.View,
                result.View));
        
            var resultStorage = new GpuScalarStorage(result);
            var gradStorage = new GpuScalarStorage(AllocateBuffer(1));
        
            return new ScalarTensor(resultStorage, [vector], Backward, gradStorage);
        
            void Backward(ScalarTensor output)
            {
                var ones = AllocateBuffer(gpuVec.TotalElements);
                Queue.Enqueue(() => VectorFillKernel(
                    (int)ones.Length,
                    ones.View,
                    ((GpuScalarStorage)output.Gradient).GpuData.View[new Index1D(0)]));
            
                AccumulateGradient(vector.Gradient, new GpuVectorStorage(ones));
            }
        }
        else
        {
            var result = Vectors.Sum(vector.Value.ToHost());
            return new ScalarTensor(
                NewCpuScalarStorage(result),
                [vector],
                Backward,
                NewCpuScalarStorage(0.0));
        
            void Backward(ScalarTensor output)
            {
                var outGrad = output.Gradient.ToHost();
                var vecLen = vector.Value.Shape[0];
                var grad = Enumerable.Repeat(outGrad, vecLen).ToArray();
                AccumulateGradient(vector.Gradient, NewCpuVectorStorage(grad));
            }
        }
    }
    #endregion
}