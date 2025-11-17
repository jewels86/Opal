using System.Numerics;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
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
    public static VectorTensor NewVector(double[] vector) => NewVector(vector, new double[vector.Length]);
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
        ArrayView1D<double, Stride1D.Dense>> VectorDivideKernel { get; private set; }
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
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>> VectorFillScalarKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>> VectorPowerKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>> VectorLogKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>> VectorSqrtKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>> VectorMaxKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>> VectorTanhKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>> VectorExpKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>> VectorScalarMaxKernel { get; set; }
    
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
    public static VectorTensorStorage BinaryOpStorage(
        VectorTensorStorage a,
        VectorTensorStorage b,
        Action<GpuVectorStorage, GpuVectorStorage, MemoryBuffer1D<double, Stride1D.Dense>> gpuKernel,
        Func<double[], double[], double[]> cpuKernel)
    {
        if (!UseGpu(a, b)) return NewCpuVectorStorage(cpuKernel(a.ToHost(), b.ToHost()));;
        var gpuA = ToGpuVector(a);
        var gpuB = ToGpuVector(b);
        var result = AllocateBuffer(gpuA.GpuData.Length);
            
        Queue.Enqueue(() => gpuKernel(gpuA, gpuB, result));
        return new GpuVectorStorage(result);
    }

    public static VectorTensorStorage BinaryOpStorage(
        VectorTensorStorage a,
        ScalarTensorStorage b,
        Action<GpuVectorStorage, GpuScalarStorage, MemoryBuffer1D<double, Stride1D.Dense>> gpuKernel,
        Func<double[], double, double[]> cpuKernel)
    {
        if (!UseGpu(a)) return NewCpuVectorStorage(cpuKernel(a.ToHost(), b.ToHost()));
        var gpuA = ToGpuVector(a);
        var gpuB = ToGpuScalar(b);
        var result = AllocateBuffer(gpuA.GpuData.Length);
        
        Queue.Enqueue(() => gpuKernel(gpuA, gpuB, result));
        return new GpuVectorStorage(result);;
    }

    public static VectorTensorStorage UnaryOpStorage(
        VectorTensorStorage vector,
        Action<GpuVectorStorage, MemoryBuffer1D<double, Stride1D.Dense>> gpuKernel,
        Func<double[], double[]> cpuKernel)
    {
        if (!UseGpu(vector)) return NewCpuVectorStorage(cpuKernel(vector.ToHost()));;
        var gpuVector = ToGpuVector(vector);
        var result = AllocateBuffer(gpuVector.GpuData.Length);
        
        Queue.Enqueue(() => gpuKernel(gpuVector, result));
        return new GpuVectorStorage(result);
    }
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

    public static VectorTensorStorage MultiplyStorage(VectorTensorStorage a, VectorTensorStorage b) =>
        BinaryOpStorage(
            a, b,
            (gpuA, gpuB, result) => 
                VectorMultiplyKernel(gpuA.GpuData.IntExtent, gpuA.GpuData.View, gpuB.GpuData.View, result.View),
            Vectors.Multiply);
    public static VectorTensorStorage DivideStorage(VectorTensorStorage a, VectorTensorStorage b) =>
        BinaryOpStorage(
            a, b,
            (gpuA, gpuB, result) => 
                VectorDivideKernel(gpuA.GpuData.IntExtent, gpuA.GpuData.View, gpuB.GpuData.View, result.View),
            Vectors.Divide);

    public static ScalarTensorStorage DotStorage(VectorTensorStorage a, VectorTensorStorage b)
    {
        if (!UseGpu(a, b)) return NewCpuScalarStorage(Vectors.Dot(a.ToHost(), b.ToHost()));
        var gpuA = ToGpuVector(a);
        var gpuB = ToGpuVector(b);
            
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
        
        return new GpuScalarStorage(result);
    }
    
    public static VectorTensorStorage ScaleVectorStorage(VectorTensorStorage vector, ScalarTensorStorage scalar) => 
        BinaryOpStorage(
            vector, scalar,
            (gpuV, gpuS, result) => 
                ScalarVectorMultiplyKernel(gpuV.GpuData.IntExtent, gpuV.GpuData.View, gpuS.GpuData.View, result.View),
            Vectors.Multiply);

    public static VectorTensorStorage NegateStorage(VectorTensorStorage vector) =>
        UnaryOpStorage(
            vector, 
            (gpuV, result) => VectorNegateKernel(gpuV.GpuData.IntExtent, gpuV.GpuData.View, result.View), 
            Vectors.Negate);

    public static VectorTensorStorage AddStorage(VectorTensorStorage a, VectorTensorStorage b) =>
        BinaryOpStorage(
            a, b,
            (gpuA, gpuB, result) =>
                VectorAddKernel(gpuA.GpuData.IntExtent, gpuA.GpuData.View, gpuB.GpuData.View, result.View),
            Vectors.Add);
    public static VectorTensorStorage SubtractStorage(VectorTensorStorage a, VectorTensorStorage b) =>
        BinaryOpStorage(
            a, b,
            (gpuA, gpuB, result) =>
                VectorSubtractKernel(gpuA.GpuData.IntExtent, gpuA.GpuData.View, gpuB.GpuData.View, result.View),
            Vectors.Subtract);

    public static VectorTensorStorage PowerStorage(VectorTensorStorage vector, ScalarTensorStorage exponent) => 
        BinaryOpStorage(
            vector, exponent,
            (gpuV, gpuS, result) => 
                VectorPowerKernel(gpuV.GpuData.IntExtent, gpuV.GpuData.View, gpuS.GpuData.View, result.View),
            (v, e) => v.Select(x => Math.Pow(x, e)).ToArray());

    public static VectorTensorStorage LogStorage(VectorTensorStorage vector) =>
        UnaryOpStorage(
            vector, 
            (gpuV, result) => VectorLogKernel(gpuV.GpuData.IntExtent, gpuV.GpuData.View, result.View), 
            v => v.Select(x => Math.Log(x)).ToArray());

    public static ScalarTensorStorage SumStorage(VectorTensorStorage vector)
    {
        if (!UseGpu(vector)) return NewCpuScalarStorage(vector.ToHost().Sum());
        var gpuVector = ToGpuVector(vector);
        var result = AllocateScalar();
        
        Queue.Enqueue(() => Accelerator.Reduce<double, AddDouble>(
            Accelerator.DefaultStream,
            gpuVector.GpuData.View,
            result.View));
        
        return new GpuScalarStorage(result);
    }

    public static VectorTensorStorage FillStorage(long length, double value)
    {
        if (!GpuAvailable) return NewCpuVectorStorage(Vectors.Fill(value, (int)length));
        var result = AllocateBuffer(length);
        Queue.Enqueue(() => VectorFillKernel(result.IntExtent, result.View, value));
        return new GpuVectorStorage(result);
    }
    public static VectorTensorStorage ExpStorage(VectorTensorStorage vector) =>
        UnaryOpStorage(
            vector, 
            (gpuV, result) => VectorExpKernel(gpuV.GpuData.IntExtent, gpuV.GpuData.View, result.View), 
            v => v.Select(Math.Exp).ToArray());
    
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
    public static VectorTensor Subtract(VectorTensor a, VectorTensor b) =>
        BinaryOp(
            a, b,
            VectorSubtractKernel, Vectors.Subtract,
            (_, _, output) =>
            {
                AccumulateGradient(a.Gradient, output.Gradient);
                AccumulateGradient(b.Gradient, NegateStorage(output.Gradient));
            });
    public static VectorTensor Multiply(VectorTensor a, VectorTensor b) =>
        BinaryOp(
            a, b,
            VectorMultiplyKernel, Vectors.Multiply, (_, _, output) =>
            {
                AccumulateGradient(a.Gradient, MultiplyStorage(b.Value, output.Gradient));
                AccumulateGradient(b.Gradient, MultiplyStorage(a.Value, output.Gradient));
            });
    public static VectorTensor Divide(VectorTensor a, VectorTensor b) =>
        BinaryOp(
            a, b,
            VectorDivideKernel, Vectors.Divide, (_, _, output) =>
            {
                AccumulateGradient(a.Gradient, DivideStorage(output.Gradient, b.Value));
                AccumulateGradient(b.Gradient, NegateStorage(DivideStorage(MultiplyStorage(output.Gradient, a.Value), MultiplyStorage(b.Value, b.Value))));
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
                output => {
                    AccumulateGradient(a.Gradient, ScaleVectorStorage(output.Gradient, scalar.Value));
                    AccumulateGradient(scalar.Gradient, DotStorage(output.Gradient, a.Value));
                },
                NewCpuVectorStorage(Vectors.Zeros(value.TotalElements)));
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
            output => {
                AccumulateGradient(a.Gradient, ScaleVectorStorage(output.Gradient, scalar.Value));
                AccumulateGradient(scalar.Gradient, DotStorage(output.Gradient, a.Value));
            },
            NewCpuVectorStorage(Vectors.Zeros(gpuAStorage.TotalElements)));
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
                Queue.Enqueue(() => VectorFillScalarKernel(
                    (int)ones.Length,
                    ones.View,
                    ((GpuScalarStorage)output.Gradient.ToGpu()).GpuData.View));
            
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

    public static VectorTensor Log(VectorTensor vector) =>
        UnaryOp(vector, VectorLogKernel,
            v => v.Select(s => Math.Log(s)).ToArray(), 
            (_, output) => AccumulateGradient(vector.Gradient, DivideStorage(output.Gradient, vector.Value)));

    public static VectorTensor Power(VectorTensor a, ScalarTensor exponent)
    {
        if (UseGpu(a.Value))
        {
            var gpuA = ToGpuVector(a.Value);
            var gpuExponent = ToGpuScalar(exponent.Value);
            var result = AllocateBuffer(gpuA.GpuData.Length);
            
            Queue.Enqueue(() => VectorPowerKernel(
                result.IntExtent,
                gpuA.GpuData.View,
                gpuExponent.GpuData.View,
                result.View));
            
            return new VectorTensor(new GpuVectorStorage(result), [a, exponent], Backwards, NewGpuVectorStorage(Vectors.Zeros((int)result.Length)));

            void Backwards(VectorTensor output)
            {
                var expMinusOne = SubtractStorage(exponent.Value, NewDefaultScalarStorage(1));
                var aPowExpMinusOne = PowerStorage(a.Value, expMinusOne);
                var gradA = MultiplyStorage(
                    ScaleVectorStorage(aPowExpMinusOne, exponent.Value),
                    output.Gradient);
                AccumulateGradient(a.Gradient, gradA);

                var logA = LogStorage(a.Value);
                var gradExponent = MultiplyStorage(
                    MultiplyStorage(output.Value, logA),
                    output.Gradient);
                
                var scalarGrad = SumStorage(gradExponent).ToHost();
                exponent.Gradient.CopyFrom(scalarGrad + exponent.Gradient.ToHost());
            }
        }
        var cpuA = a.Value.ToHost();
        return new VectorTensor(
            NewCpuVectorStorage(cpuA.Select(x => Math.Pow(x, exponent.Value.ToHost())).ToArray()),
            [a, exponent], output =>
            {
                var cpuA2 = a.Value.ToHost();
                var cpuExponent = exponent.Value.ToHost();
                var outGrad = output.Gradient.ToHost();
                var outValue = output.Value.ToHost();
    
                var gradA = cpuA2.Select((val, i) => 
                    cpuExponent * Math.Pow(val, cpuExponent - 1) * outGrad[i]).ToArray();
                AccumulateGradient(a.Gradient, NewCpuVectorStorage(gradA));
    
                var gradExponent = cpuA2.Select((val, i) => 
                    outValue[i] * Math.Log(val) * outGrad[i]).Sum();
                exponent.Gradient.CopyFrom(gradExponent + exponent.Gradient.ToHost());
            }, NewCpuVectorStorage(Vectors.Zeros(cpuA.Length)));
    }
    public static VectorTensor Tanh(VectorTensor vector) =>
        UnaryOp(
            vector,
            VectorTanhKernel,
            v => v.Select(Math.Tanh).ToArray(),
            (input, output) =>
            {
                var tanhSquared = MultiplyStorage(output.Value, output.Value);
                var oneMinusTanhSquared = SubtractStorage(
                    FillStorage(vector.Value.TotalElements, 1.0),
                    tanhSquared);
                var grad = MultiplyStorage(oneMinusTanhSquared, output.Gradient);
                AccumulateGradient(input.Gradient, grad);
            });

    public static VectorTensor Fill(long length, double value, double gradValue) => NewVector(
        FillStorage(length, value), null,
        _ => { }, FillStorage(length, gradValue));
    
    public static VectorTensor Exp(VectorTensor vector) =>
        UnaryOp(
            vector,
            VectorExpKernel,
            v => v.Select(Math.Exp).ToArray(),
            (input, output) =>
            {
                var grad = MultiplyStorage(output.Value, output.Gradient);
                AccumulateGradient(input.Gradient, grad);
            });
    public static VectorTensor ReLuVector(VectorTensor x) =>
        UnaryOp(
            x,
            (i, v, r) => 
                VectorScalarMaxKernel(i, v, ToGpuScalar(NewGpuScalarStorage(0.0)).GpuData.View, r), // GPU
            v => v.Select(val => Math.Max(0, val)).ToArray(),
            (input, output) =>
            {
                var xVal = input.Value.ToHost();
                var mask = xVal.Select(v => v > 0 ? 1.0 : 0.0).ToArray();
                var grad = MultiplyStorage(
                    NewDefaultVectorStorage(mask), 
                    output.Gradient);
                AccumulateGradient(input.Gradient, grad);
            });
    #endregion
}