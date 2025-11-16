using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using Opal.Mathematics;

namespace Opal.Autograd;

public static partial class Operations
{
    #region Vector Tensor Helpers
    public static ITensorStorage<double[]> NewCpuVectorStorage(double[] vector) =>
        new CpuStorage<double[]>(vector, [vector.Length], vector.Length);

    public static ITensorStorage<double[]> NewGpuVectorStorage(double[] vector)
    {
        var buffer = Accelerator.Allocate1D<double>(vector.Length);
        buffer.CopyFromCPU(vector);
        return new GpuVectorStorage(buffer);
    }
    
    public static ITensorStorage<double[]> NewDefaultVectorStorage(double[] vector) => 
        GpuAvailable ? NewGpuVectorStorage(vector) : NewCpuVectorStorage(vector);

    public static VectorTensor NewVector(ITensorStorage<double[]> storage, List<object>? inputs, Action<Tensor<ITensorStorage<double[]>>> backwards,
        ITensorStorage<double[]> gradient) =>
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
    public static VectorTensor BinaryOp(
        VectorTensor a,
        VectorTensor b,
        Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>> gpuKernel,
        Func<double[], double[], double[]> cpuFallback,
        Action<VectorTensor, VectorTensor, VectorTensor> gradientFn)
    {
        if (GpuAvailable && 
            (a.Value is GpuVectorStorage || b.Value is GpuVectorStorage))
        {
            var gpuA = (a.Value as GpuVectorStorage) ?? 
                       (GpuVectorStorage)a.Value.ToGpu();
            var gpuB = (b.Value as GpuVectorStorage) ?? 
                       (GpuVectorStorage)b.Value.ToGpu();
        
            var resultBuffer = Accelerator.Allocate1D<double>(gpuA.GpuData.Length);
        
            Queue.Enqueue(() => gpuKernel((int)gpuA.GpuData.Length, gpuA.GpuData.View, gpuB.GpuData.View, resultBuffer.View));
        
            var resultStorage = new GpuVectorStorage(resultBuffer);
            var gradStorage = new GpuVectorStorage(Accelerator.Allocate1D<double>((int)resultBuffer.Length));
        
            return new VectorTensor(resultStorage, [a, b], 
                output => gradientFn(a, b, (VectorTensor)output), gradStorage);
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
        if (GpuAvailable && vector.Value is GpuVectorStorage gpuVector)
        {
            var resultBuffer = Accelerator.Allocate1D<double>(gpuVector.GpuData.Length);
            Queue.Enqueue(() => gpuKernel((int)gpuVector.GpuData.Length, gpuVector.GpuData.View, resultBuffer.View));
            var resultStorage = new GpuVectorStorage(resultBuffer);
            var gradStorage = new GpuVectorStorage(Accelerator.Allocate1D<double>((int)resultBuffer.Length));
            return new VectorTensor(resultStorage, [vector], output => gradientFn(vector, (VectorTensor)output), gradStorage);;
        }
        var result = cpuFallback(vector.Value.ToHost());
        return NewVector(
            NewCpuVectorStorage(result), 
            [vector], 
            output => gradientFn(vector, (VectorTensor)output), 
            NewCpuVectorStorage(new double[result.Length]));
    }
    #endregion
    #region Storage Helpers
    public static void AccumulateGradient(
        ITensorStorage<double[]> gradient,
        ITensorStorage<double[]> incomingGrad)
    {
        if (GpuAvailable &&
            (gradient is GpuVectorStorage || incomingGrad is GpuVectorStorage))
        {
            var gpuGrad = (gradient as GpuVectorStorage) ??
                          (GpuVectorStorage)gradient.ToGpu();
            var gpuIncoming = (incomingGrad as GpuVectorStorage) ??
                              (GpuVectorStorage)incomingGrad.ToGpu();

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

    public static ITensorStorage<double[]> MultiplyStorage(ITensorStorage<double[]> a, ITensorStorage<double[]> b)
    {
        if (GpuAvailable &&
            (a is GpuVectorStorage || b is GpuVectorStorage))
        {
            var gpuA = a as GpuVectorStorage ?? (GpuVectorStorage)a.ToGpu();
            var gpuB = b as GpuVectorStorage ?? (GpuVectorStorage)b.ToGpu();
            
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
    
    public static ITensorStorage<double[]> ScaleVectorStorage(
        ITensorStorage<double[]> vector, 
        ITensorStorage<double> scalar)
    {
        if (vector is GpuVectorStorage gpuVec && scalar is GpuScalarStorage gpuScalar)
        {
            var result = Accelerator.Allocate1D<double>(gpuVec.GpuData.Length);
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

    public static ITensorStorage<double[]> NegateStorage(ITensorStorage<double[]> vector)
    {
        if (GpuAvailable && vector is GpuVectorStorage gpuVector)
        {
            var result = Accelerator.Allocate1D<double>(gpuVector.GpuData.Length);
            Queue.Enqueue(() => VectorNegateKernel(
                (int)gpuVector.GpuData.Length,
                gpuVector.GpuData.View,
                result.View));
            return new GpuVectorStorage(result);;
        }
        var vecData = vector.ToHost();
        return NewCpuVectorStorage(Vectors.Negate(vecData));
    }

    public static ITensorStorage<double[]> AddStorage(ITensorStorage<double[]> a, ITensorStorage<double[]> b)
    {
        if (!GpuAvailable || (a is not GpuVectorStorage && b is not GpuVectorStorage)) return NewCpuVectorStorage(Vectors.Add(a.ToHost(), b.ToHost()));
        var gpuA = (a as GpuVectorStorage) ?? (GpuVectorStorage)a.ToGpu();
        var gpuB = (b as GpuVectorStorage) ?? (GpuVectorStorage)b.ToGpu();
        var result = Accelerator.Allocate1D<double>(gpuA.GpuData.Length);
            
        Queue.Enqueue(() => VectorAddKernel(
            (int)gpuA.GpuData.Length,
            gpuA.GpuData.View,
            gpuB.GpuData.View,
            result.View));
        return new GpuVectorStorage(result);
    }
    public static ITensorStorage<double[]> SubtractStorage(ITensorStorage<double[]> a, ITensorStorage<double[]> b)
    {
        if (!GpuAvailable || (a is not GpuVectorStorage && b is not GpuVectorStorage)) return NewCpuVectorStorage(Vectors.Subtract(a.ToHost(), b.ToHost()));
        var gpuA = (a as GpuVectorStorage) ?? (GpuVectorStorage)a.ToGpu();
        var gpuB = (b as GpuVectorStorage) ?? (GpuVectorStorage)b.ToGpu();
        var result = Accelerator.Allocate1D<double>(gpuA.GpuData.Length);
            
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
        if (a.Value is GpuVectorStorage gpuA && b.Value is GpuVectorStorage gpuB)
        {
            var product = Accelerator.Allocate1D<double>(gpuA.GpuData.Length);
            Queue.Enqueue(() => VectorMultiplyKernel(
                (int)gpuA.GpuData.Length,
                gpuA.GpuData.View,
                gpuB.GpuData.View,
                product.View));
        
            var result = Accelerator.Allocate1D<double>(1);
            Queue.Enqueue(() => 
            {
                Accelerator.Reduce<double, AddDouble>(
                    Accelerator.DefaultStream,
                    product.View,
                    result.View);
            });
        
            var resultStorage = new GpuScalarStorage(result);
            var gradStorage = new GpuScalarStorage(Accelerator.Allocate1D<double>(1));
        
            return new ScalarTensor(resultStorage, [a, b], Backward, gradStorage);
        
            void Backward(Tensor<ITensorStorage<double>> output)
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
                AccumulateGradient(a.Gradient, 
                    NewCpuVectorStorage(Vectors.Multiply(b.Value.ToHost(), outGrad)));
                AccumulateGradient(b.Gradient, 
                    NewCpuVectorStorage(Vectors.Multiply(a.Value.ToHost(), outGrad)));
            }
        }
    }
    public static VectorTensor Concat(VectorTensor a, VectorTensor b)
    {
        if (a.Value is GpuVectorStorage gpuA && b.Value is GpuVectorStorage gpuB)
        {
            int aLength = (int)gpuA.GpuData.Length;
            int bLength = (int)gpuB.GpuData.Length;
            int totalLength = aLength + bLength;
            
            var result = Accelerator.Allocate1D<double>(totalLength);
            
            Queue.Enqueue(() => GpuKernels.VectorConcatKernel(
                totalLength,
                gpuA.GpuData.View,
                gpuB.GpuData.View,
                result.View,
                aLength));
            
            var resultStorage = new GpuVectorStorage(result);
            var gradStorage = new GpuVectorStorage(
                Accelerator.Allocate1D<double>(totalLength));
            
            return new VectorTensor(resultStorage, [a, b], Backward, gradStorage);
            
            void Backward(VectorTensor output)
            {
                var outputVec = output;
    
                if (outputVec.Gradient is GpuVectorStorage gpuOutGrad &&
                    a.Gradient is GpuVectorStorage gpuGradA &&
                    b.Gradient is GpuVectorStorage gpuGradB)
                {
                    var tempA = Accelerator.Allocate1D<double>(aLength);
                    Queue.Enqueue(() => VectorSliceKernel(aLength, gpuOutGrad.GpuData.View, tempA.View, 0));
                    AccumulateGradient(a.Gradient, new GpuVectorStorage(tempA));
        
                    var tempB = Accelerator.Allocate1D<double>(bLength);
                    Queue.Enqueue(() => VectorSliceKernel(bLength, gpuOutGrad.GpuData.View, tempB.View, aLength));
                    AccumulateGradient(b.Gradient, new GpuVectorStorage(tempB));
                }
                else
                {
                    var outGrad = outputVec.Gradient.ToHost();
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
            
            void Backward(Tensor<ITensorStorage<double[]>> output)
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
        if (!GpuAvailable || a.Value is not GpuVectorStorage gpuAStorage)
        {
            var value = NewCpuVectorStorage(Vectors.Multiply(a.Value.ToHost(), scalar.Value.ToHost()));
            return NewVector(
                value, [a, scalar], 
                (output) => AccumulateGradient(a.Gradient, ScaleVectorStorage(output.Gradient, scalar.Gradient)),
                NewCpuVectorStorage(Vectors.Ones(value.TotalElements)));
        }
        var gpuScalarStorage = (scalar.Value as GpuScalarStorage) ?? (GpuScalarStorage)scalar.Value.ToGpu();
        var result = Accelerator.Allocate1D<double>(gpuAStorage.TotalElements);
        
        Queue.Enqueue(() => ScalarVectorMultiplyKernel(
            gpuAStorage.TotalElements,
            gpuAStorage.GpuData.View,
            gpuScalarStorage.GpuData.View,
            result.View));
        
        return NewVector(
            new GpuVectorStorage(result), [a, scalar], 
            (output) => AccumulateGradient(a.Gradient, ScaleVectorStorage(output.Gradient, scalar.Gradient)),
            NewCpuVectorStorage(Vectors.Ones(gpuAStorage.TotalElements)));
    }

    public static VectorTensor Negate(VectorTensor a) => UnaryOp(
        a, VectorNegateKernel, Vectors.Negate, 
        (_, output) => AccumulateGradient(a.Gradient, NegateStorage(output.Gradient)));
    
    public static ScalarTensor Sum(VectorTensor vector)
    {
        if (GpuAvailable && vector.Value is GpuVectorStorage gpuVec)
        {
            var result = Accelerator.Allocate1D<double>(1);
            Queue.Enqueue(() => 
            {
                Accelerator.Reduce<double, AddDouble>(
                    Accelerator.DefaultStream,
                    gpuVec.GpuData.View,
                    result.View);
            });
        
            var resultStorage = new GpuScalarStorage(result);
            var gradStorage = new GpuScalarStorage(Accelerator.Allocate1D<double>(1));
        
            return new ScalarTensor(resultStorage, [vector], Backward, gradStorage);
        
            void Backward(Tensor<ITensorStorage<double>> output)
            {
                var ones = Accelerator.Allocate1D<double>((int)gpuVec.GpuData.Length);
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
        
            void Backward(Tensor<ITensorStorage<double>> output)
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