using System.Diagnostics;
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
        var buffer = Controller.Get(vector.Length);
        buffer.CopyFromCPU(vector);
        return new GpuVectorStorage(buffer);
    }
    public static VectorTensorStorage NewDefaultVectorStorage(double[] vector) => 
        GpuAvailable ? NewGpuVectorStorage(vector) : NewCpuVectorStorage(vector);
    public static VectorTensor NewVector(VectorTensorStorage storage, List<object>? inputs, Action<VectorTensor> backwards,
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
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>> VectorCopyKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>> VectorMaskedMultiplyKernel { get; private set; }
    
    #endregion
    #region Helpers
    public static bool UseGpu(params VectorTensorStorage[] storages) => storages.Any(s => s is GpuVectorStorage) && GpuAvailable;
    public static GpuVectorStorage ToGpuVector(VectorTensorStorage storage) => storage as GpuVectorStorage ?? (GpuVectorStorage)storage.ToGpu();
    public static MemoryBuffer1D<double, Stride1D.Dense> AllocateBuffer(long length) => Controller.Get((int)length);
    public static MemoryBuffer1D<double, Stride1D.Dense> AllocateTemp(long length) => Controller.GetTemp((int)length);
    
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
        
            gpuKernel((int)gpuA.GpuData.Length, gpuA.GpuData.View, gpuB.GpuData.View, resultBuffer.View);
        
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
            
            gpuKernel((int)gpuVector.GpuData.Length, gpuVector.GpuData.View, resultBuffer.View);
            
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

    public static void AccumulateInto(
        ArrayView1D<double, Stride1D.Dense> gradient,
        Action<ArrayView1D<double, Stride1D.Dense>> computeIntoTemp,
        bool subtract = false)
    {
        var temp = AllocateTemp(gradient.Length);
        computeIntoTemp(temp.View);
        if (!subtract) VectorAddKernel(temp.IntExtent, gradient, temp.View, gradient);
        else VectorSubtractKernel(temp.IntExtent, gradient, temp.View, gradient);
    }
    public static void BinaryGradientOp(
        VectorTensor a,
        VectorTensor b,
        VectorTensor output,
        Action<ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>
            gpuGradientFn,
        Action<VectorTensorStorage, VectorTensorStorage, VectorTensorStorage, VectorTensorStorage, VectorTensorStorage> cpuGradientFn)
    {
        if (UseGpu(a.Gradient, b.Gradient, output.Gradient))
        {
            var gpuAGrad = ToGpuVector(a.Gradient);
            var gpuBGrad = ToGpuVector(b.Gradient);
            var gpuOut = ToGpuVector(output.Gradient);
            var gpuAVal = ToGpuVector(a.Value);
            var gpuBVal = ToGpuVector(b.Value);
        
            gpuGradientFn(gpuAGrad.GpuData.View, gpuBGrad.GpuData.View, 
                gpuOut.GpuData.View, gpuAVal.GpuData.View, gpuBVal.GpuData.View);
        }
        else
            cpuGradientFn(a.Gradient, b.Gradient, output.Gradient, a.Value, b.Value);
    }

    public static void UnaryGradientOp(
        VectorTensor vector,
        VectorTensor output,
        Action<ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>> gpuGradientFn,
        Action<VectorTensorStorage, VectorTensorStorage, VectorTensorStorage> cpuGradientFn)
    {
        if (UseGpu(vector.Gradient, output.Gradient))
        {
            var gpuGrad = ToGpuVector(vector.Gradient);
            var gpuOut = ToGpuVector(output.Gradient);
            var gpuVal = ToGpuVector(vector.Value);
        
            gpuGradientFn(gpuGrad.GpuData.View, gpuOut.GpuData.View, gpuVal.GpuData.View);
        }
        else
            cpuGradientFn(vector.Gradient, output.Gradient, vector.Value);
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
            
        gpuKernel(gpuA, gpuB, result);
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
        
        gpuKernel(gpuA, gpuB, result);
        return new GpuVectorStorage(result);
    }

    public static VectorTensorStorage UnaryOpStorage(
        VectorTensorStorage vector,
        Action<GpuVectorStorage, MemoryBuffer1D<double, Stride1D.Dense>> gpuKernel,
        Func<double[], double[]> cpuKernel)
    {
        if (!UseGpu(vector)) return NewCpuVectorStorage(cpuKernel(vector.ToHost()));;
        var gpuVector = ToGpuVector(vector);
        var result = AllocateBuffer(gpuVector.GpuData.Length);
        
        gpuKernel(gpuVector, result);
        return new GpuVectorStorage(result);
    }
    public static void AccumulateGradient(VectorTensorStorage gradient, VectorTensorStorage incomingGrad)
    {
        if (UseGpu(gradient, incomingGrad))
        {
            var gpuGrad = ToGpuVector(gradient);
            var gpuIncoming = ToGpuVector(incomingGrad);

            VectorAddKernel(
                (int)gpuGrad.GpuData.Length,
                gpuGrad.GpuData.View,
                gpuIncoming.GpuData.View,
                gpuGrad.GpuData.View);
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
            
        var product = AllocateTemp(gpuA.GpuData.Length);
            
        VectorMultiplyKernel(
            (int)gpuA.GpuData.Length,
            gpuA.GpuData.View,
            gpuB.GpuData.View,
            product.View);
        
        var result = AllocateScalar();
        Accelerator.Reduce<double, AddDouble>(
            Accelerator.DefaultStream,
            product.View,
            result.View);
        
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
        
        Accelerator.Reduce<double, AddDouble>(
            Accelerator.DefaultStream,
            gpuVector.GpuData.View,
            result.View);
        
        return new GpuScalarStorage(result);
    }

    public static VectorTensorStorage FillStorage(long length, double value)
    {
        if (!GpuAvailable) return NewCpuVectorStorage(Vectors.Fill(value, (int)length));
        var result = AllocateBuffer(length);
        VectorFillKernel(result.IntExtent, result.View, value);
        return new GpuVectorStorage(result);
    }

    public static void FillStorage(VectorTensorStorage storage, double value)
    {
        if (!UseGpu(storage)) ((CpuStorage<double[]>)storage).Data = Vectors.Fill(value, storage.TotalElements);
        var gpuStorage = ToGpuVector(storage);
        VectorFillKernel(
            gpuStorage.GpuData.IntExtent, 
            gpuStorage.GpuData.View, 
            value);
    }
    public static VectorTensorStorage ExpStorage(VectorTensorStorage vector) =>
        UnaryOpStorage(
            vector, 
            (gpuV, result) => VectorExpKernel(gpuV.GpuData.IntExtent, gpuV.GpuData.View, result.View), 
            v => v.Select(Math.Exp).ToArray());
    
    #endregion
    #region Memory Ops
    public static void CopyMemory(ArrayView1D<double, Stride1D.Dense> source, ArrayView1D<double, Stride1D.Dense> destination) => 
        VectorCopyKernel(source.IntExtent, source, destination);
    public static void DotMemory(ArrayView1D<double, Stride1D.Dense> a, ArrayView1D<double, Stride1D.Dense> b, ArrayView1D<double, Stride1D.Dense> result)
    {
        var product = AllocateTemp(a.Length);
            
        VectorMultiplyKernel(
            a.IntExtent,
            a,
            b,
            product.View);
        
        Accelerator.Reduce<double, AddDouble>(
            Accelerator.DefaultStream,
            product.View,
            result);
    }

    public static void AddMemory(ArrayView1D<double, Stride1D.Dense> a, ArrayView1D<double, Stride1D.Dense> b, ArrayView1D<double, Stride1D.Dense> result) => 
        VectorAddKernel(a.IntExtent, a, b, result);
    public static void SubtractMemory(ArrayView1D<double, Stride1D.Dense> a, ArrayView1D<double, Stride1D.Dense> b, ArrayView1D<double, Stride1D.Dense> result) => 
        VectorSubtractKernel(a.IntExtent, a, b, result);
    public static void MultiplyMemory(ArrayView1D<double, Stride1D.Dense> a, ArrayView1D<double, Stride1D.Dense> b, ArrayView1D<double, Stride1D.Dense> result) => 
        VectorMultiplyKernel(a.IntExtent, a, b, result);
    public static void DivideMemory(ArrayView1D<double, Stride1D.Dense> a, ArrayView1D<double, Stride1D.Dense> b, ArrayView1D<double, Stride1D.Dense> result) => 
        VectorDivideKernel(a.IntExtent, a, b, result);
    public static void NegateMemory(ArrayView1D<double, Stride1D.Dense> a, ArrayView1D<double, Stride1D.Dense> result) => 
        VectorNegateKernel(a.IntExtent, a, result);
    public static void LogMemory(ArrayView1D<double, Stride1D.Dense> a, ArrayView1D<double, Stride1D.Dense> result) => 
        VectorLogKernel(a.IntExtent, a, result);
    public static void FillMemory(ArrayView1D<double, Stride1D.Dense> storage, double value) => 
        VectorFillKernel(storage.IntExtent, storage, value);
    public static void FillMemory(ArrayView1D<double, Stride1D.Dense> storage, ArrayView1D<double, Stride1D.Dense> scalar) =>
        VectorFillScalarKernel(storage.IntExtent, storage, scalar);
    public static void ExpMemory(ArrayView1D<double, Stride1D.Dense> storage, ArrayView1D<double, Stride1D.Dense> result) => 
        VectorExpKernel(storage.IntExtent, storage, result);
    public static void PowerMemory(ArrayView1D<double, Stride1D.Dense> storage, ArrayView1D<double, Stride1D.Dense> exponent, ArrayView1D<double, Stride1D.Dense> result) => 
        VectorPowerKernel(storage.IntExtent, storage, exponent, result);
    public static void ScaleMemory(ArrayView1D<double, Stride1D.Dense> storage, ArrayView1D<double, Stride1D.Dense> scalar, ArrayView1D<double, Stride1D.Dense> result) => 
        VectorScalarMaxKernel(storage.IntExtent, storage, scalar, result);

    public static void SumMemory(ArrayView1D<double, Stride1D.Dense> storage, ArrayView1D<double, Stride1D.Dense> result) =>
        Accelerator.Reduce<double, AddDouble>(
            Accelerator.DefaultStream,
            storage,
            result);
    public static void MaxMemory(ArrayView1D<double, Stride1D.Dense> storage, ArrayView1D<double, Stride1D.Dense> scalar, ArrayView1D<double, Stride1D.Dense> result) =>
        VectorMaxKernel(storage.IntExtent, storage, scalar, result);
    public static void MaskedMultiplyMemory(ArrayView1D<double, Stride1D.Dense> storage, ArrayView1D<double, Stride1D.Dense> values, ArrayView1D<double, Stride1D.Dense> result) =>
        VectorMaskedMultiplyKernel(storage.IntExtent, storage, values, result);
    #endregion
    #region Operations
    #region Simple Ops
    public static VectorTensor Add(VectorTensor a, VectorTensor b) => 
        BinaryOp(
            a, b, 
            VectorAddKernel, Vectors.Add, (_, _, output) => BinaryGradientOp(
                a, b, output,
                (aGrad, bGrad, outGrad, _, _) =>
                {
                    AccumulateInto(aGrad, temp => VectorCopyKernel(outGrad.IntExtent, outGrad, temp));
                    AccumulateInto(bGrad, temp => VectorCopyKernel(outGrad.IntExtent, outGrad, temp));
                }, (aGrad, bGrad, outGrad, _, _) =>
                {
                    AccumulateGradient(aGrad, outGrad);
                    AccumulateGradient(bGrad, outGrad);
                }));
    public static VectorTensor Subtract(VectorTensor a, VectorTensor b) =>
        BinaryOp(
            a, b,
            VectorSubtractKernel, Vectors.Subtract, (_, _, output) => BinaryGradientOp(
                a, b, output,
                (aGrad, bGrad, outGrad, _, _) =>
                {
                    AccumulateInto(aGrad, temp => VectorCopyKernel(outGrad.IntExtent, outGrad, temp));
                    AccumulateInto(bGrad, temp => VectorNegateKernel(outGrad.IntExtent, outGrad, temp));
                }, (aGrad, bGrad, outGrad, aVal, bVal) =>
                {
                    AccumulateGradient(aGrad, outGrad);
                    AccumulateGradient(bGrad, NegateStorage(outGrad));
                }));
    public static VectorTensor Multiply(VectorTensor a, VectorTensor b) =>
        BinaryOp(
            a, b,
            VectorMultiplyKernel, Vectors.Multiply, (_, _, output) => BinaryGradientOp(
                a, b, output,
                (aGrad, bGrad, outGrad, aVal, bVal) =>
                {
                    AccumulateInto(aGrad, temp => VectorMultiplyKernel(
                        (int)bVal.Length, bVal, outGrad, temp));
                    AccumulateInto(bGrad, temp => VectorMultiplyKernel(
                        (int)aVal.Length, aVal, outGrad, temp));
                },
                (aGrad, bGrad, outGrad, aVal, bVal) =>
                {
                    AccumulateGradient(aGrad, MultiplyStorage(bVal, outGrad));
                    AccumulateGradient(bGrad, MultiplyStorage(aVal, outGrad));
                }));
    public static VectorTensor Divide(VectorTensor a, VectorTensor b) =>
        BinaryOp(
            a, b,
            VectorDivideKernel, Vectors.Divide, (_, _, output) => BinaryGradientOp(
                a, b, output,
                (aGrad, bGrad, outGrad, aVal, bVal) =>
                {
                    AccumulateInto(aGrad, temp => VectorDivideKernel(outGrad.IntExtent, outGrad, bVal, temp));
                    AccumulateInto(bGrad, temp =>
                    {
                        var (mul1, mul2, div) =
                            (AllocateTemp(aVal.Length), AllocateTemp(bVal.Length), AllocateTemp(aVal.Length));
                        VectorMultiplyKernel(outGrad.IntExtent, outGrad, aVal, mul1);
                        VectorMultiplyKernel(bVal.IntExtent, bVal, bVal, mul2);
                        VectorDivideKernel(outGrad.IntExtent, mul1, mul2, div);
                        VectorNegateKernel(outGrad.IntExtent, div, temp);
                    });
                }, (aGrad, bGrad, outGrad, aVal, bVal) =>
                {
                    AccumulateGradient(aGrad, DivideStorage(outGrad, bVal));
                    AccumulateGradient(bGrad, NegateStorage(DivideStorage(MultiplyStorage(outGrad, aVal), MultiplyStorage(bVal, bVal))));
                }));
    public static VectorTensor Negate(VectorTensor a) => UnaryOp(
        a, VectorNegateKernel, Vectors.Negate,
        (_, output) => UnaryGradientOp(
            a, output, (gpuGrad, gpuOut, gpuVal) => 
                AccumulateInto(gpuGrad, temp => VectorCopyKernel(gpuOut.IntExtent, gpuOut, temp), true),
            (_, _, outGrad) => AccumulateGradient(a.Gradient, NegateStorage(outGrad))));
    public static VectorTensor Log(VectorTensor vector) =>
        UnaryOp(vector, VectorLogKernel,
            v => v.Select(s => Math.Log(s)).ToArray(), 
            (_, output) => UnaryGradientOp(
                vector, output, (gpuGrad, gpuOut, gpuVal) => 
                    AccumulateInto(gpuGrad, temp => DivideMemory(gpuGrad, gpuOut, temp)),
                (_, _, outGrad) => AccumulateGradient(vector.Gradient, DivideStorage(outGrad, outGrad))));
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
                if (UseGpu(output.Gradient))
                {
                    var gpuOutGrad = ToGpuVector(output.Gradient);
                    var gpuOutVal = ToGpuVector(output.Value);
                    var gpuInputGrad = ToGpuVector(input.Gradient);
                    
                    AccumulateInto(gpuInputGrad, temp => MultiplyMemory(gpuOutGrad, gpuOutVal, temp));;
                }
                else
                    AccumulateGradient(input.Gradient, MultiplyStorage(output.Value, output.Gradient));
            });
    public static VectorTensor ReLuVector(VectorTensor x) =>
        UnaryOp(
            x,
            (i, v, r) => 
                VectorScalarMaxKernel(i, v, ToGpuScalar(Zero).GpuData.View, r),
            v => v.Select(val => Math.Max(0, val)).ToArray(),
            (input, output) =>
            {
                if (UseGpu(output.Gradient))
                {
                    var gpuInputVal = ToGpuVector(input.Value);
                    var gpuInputGrad = ToGpuVector(input.Gradient);
                    var gpuOutGrad = ToGpuVector(output.Gradient);
                    var gpuOutVal = ToGpuVector(output.Value);
                    
                    var (gpuMask, ones) = (AllocateTemp(gpuInputVal.GpuData.Length), AllocateTemp(gpuInputVal.GpuData.Length));
                    MaskedMultiplyMemory(gpuOutVal, ones, gpuMask);
                    AccumulateInto(gpuInputGrad, temp => MultiplyMemory(gpuMask, gpuOutGrad, temp));
                }
                var xVal = input.Value.ToHost();
                var mask = xVal.Select(v => v > 0 ? 1.0 : 0.0).ToArray();
                var grad = MultiplyStorage(
                    NewDefaultVectorStorage(mask), 
                    output.Gradient);
                AccumulateGradient(input.Gradient, grad);
            });
    public static VectorTensor Tanh(VectorTensor vector) =>
        UnaryOp(
            vector,
            VectorTanhKernel,
            v => v.Select(Math.Tanh).ToArray(),
            (input, output) =>
            {
                if (UseGpu(output.Gradient, input.Gradient))
                {
                    var gpuOutGrad = ToGpuVector(output.Gradient);
                    var gpuOutVal = ToGpuVector(output.Value);
                    var gpuInputGrad = ToGpuVector(input.Gradient);
                    
                    var (gpuTanhSquared, ones, gpuOneMinusTanhSquared) =
                        (AllocateTemp(gpuOutGrad.GpuData.Length), AllocateTemp(gpuOutGrad.GpuData.Length), AllocateTemp(gpuOutGrad.GpuData.Length));
                    
                    FillMemory(ones, 1.0);
                    MultiplyMemory(gpuOutVal, gpuOutVal, gpuTanhSquared);
                    SubtractMemory(ones, gpuTanhSquared, gpuOneMinusTanhSquared);
                    AccumulateInto(gpuInputGrad, temp => MultiplyMemory(gpuOutGrad, gpuOneMinusTanhSquared, temp));
                    
                }
                var tanhSquared = MultiplyStorage(output.Value, output.Value);
                var oneMinusTanhSquared = SubtractStorage(
                    FillStorage(vector.Value.TotalElements, 1.0),
                    tanhSquared);
                var grad = MultiplyStorage(oneMinusTanhSquared, output.Gradient);
                AccumulateGradient(input.Gradient, grad);
            });
    #endregion
    public static ScalarTensor Dot(VectorTensor a, VectorTensor b)
    {
        if (UseGpu(a.Value, b.Value))
        {
            var gpuA = ToGpuVector(a.Value);
            var gpuB = ToGpuVector(b.Value);
            var result = AllocateScalar();
            DotMemory(gpuA, gpuB, result);
        
            var resultStorage = new GpuScalarStorage(result);
            var gradStorage = new GpuScalarStorage(AllocateScalar());
        
            return new ScalarTensor(resultStorage, [a, b], Backward, gradStorage);

            void Backward(ScalarTensor output)
            {
                var gpuOutGrad = ToGpuScalar(output.Gradient);
                var gpuAGrad = ToGpuVector(a.Gradient);
                var gpuAVal = ToGpuVector(a.Value);
                var gpuBGrad = ToGpuVector(b.Gradient);
                var gpuBVal = ToGpuVector(b.Value);
                
                AccumulateInto(gpuAGrad, temp => ScaleMemory(gpuBVal, gpuOutGrad, temp));
                AccumulateInto(gpuBGrad, temp => ScaleMemory(gpuAVal, gpuOutGrad, temp));
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
            VectorConcatKernel(
                result.IntExtent,
                gpuA.GpuData.View,
                gpuB.GpuData.View,
                result.View,
                aLength);
            
            var resultStorage = new GpuVectorStorage(result);
            var gradStorage = new GpuVectorStorage(AllocateBuffer(totalLength));
            
            return new VectorTensor(resultStorage, [a, b], Backward, gradStorage);
            
            void Backward(VectorTensor output) =>
                BinaryGradientOp(a, b, output,
                    (aGrad, bGrad, outGrad, _, _) =>
                    {
                        AccumulateInto(aGrad, temp => VectorSliceKernel(aLength, outGrad, temp, 0));
                        AccumulateInto(bGrad, temp => VectorSliceKernel(bLength, outGrad, temp, aLength));
                    },
                    (aGrad, bGrad, outGrad, _, _) =>
                    {
                        var outGradValue = outGrad.ToHost();
                        var gradA = outGradValue[..aLength];
                        var gradB = outGradValue[aLength..];
        
                        AccumulateGradient(aGrad, NewCpuVectorStorage(gradA));
                        AccumulateGradient(bGrad, NewCpuVectorStorage(gradB));
                    });
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
                
                AccumulateGradient(a.Gradient, NewCpuVectorStorage(gradA));
                AccumulateGradient(b.Gradient, NewCpuVectorStorage(gradB));
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
        
        ScalarVectorMultiplyKernel(
            gpuAStorage.TotalElements,
            gpuAStorage.GpuData.View,
            gpuScalarStorage.GpuData.View,
            result.View);
        
        return new VectorTensor(
            new GpuVectorStorage(result), [a, scalar], 
            output => {
                if (UseGpu(a.Gradient))
                {
                    var gpuAGrad = ToGpuVector(a.Gradient);
                    var gpuAVal = ToGpuVector(a.Value);
                    var gpuScalarGrad = ToGpuScalar(scalar.Gradient);
                    var gpuScalarVal = ToGpuScalar(scalar.Value);
                    var gpuOutGrad = ToGpuVector(output.Gradient);
                    
                    AccumulateInto(gpuAGrad, temp => ScalarVectorMultiplyKernel(gpuOutGrad.GpuData.IntExtent, gpuOutGrad, gpuScalarVal, temp));
                    AccumulateInto(gpuScalarGrad, temp => DotMemory(gpuOutGrad, gpuAVal, temp));
                }
                else
                {
                    AccumulateGradient(a.Gradient, ScaleVectorStorage(output.Gradient, scalar.Value));
                    AccumulateGradient(scalar.Gradient, DotStorage(output.Gradient, a.Value));
                }
            },
            new GpuVectorStorage(AllocateBuffer(gpuAStorage.TotalElements)));
    }
    public static ScalarTensor Sum(VectorTensor vector)
    {
        if (UseGpu(vector.Value))
        {
            var gpuVec = ToGpuVector(vector.Value);
            var result = AllocateScalar();
            SumMemory(gpuVec, result);
        
            var resultStorage = new GpuScalarStorage(result);
            var gradStorage = new GpuScalarStorage(AllocateBuffer(1));
        
            return new ScalarTensor(resultStorage, [vector], Backward, gradStorage);
        
            void Backward(ScalarTensor output)
            {
                var gpuGrad = ToGpuVector(vector.Gradient);
                var gpuOutGrad = ToGpuScalar(output.Gradient);
            
                AccumulateInto(gpuGrad, temp => FillMemory(temp, gpuOutGrad));
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
    public static VectorTensor Power(VectorTensor a, ScalarTensor exponent)
    {
        if (UseGpu(a.Value))
        {
            var gpuA = ToGpuVector(a.Value);
            var gpuExponent = ToGpuScalar(exponent.Value);
            var result = AllocateBuffer(gpuA.GpuData.Length);
            
            VectorPowerKernel(
                result.IntExtent,
                gpuA.GpuData.View,
                gpuExponent.GpuData.View,
                result.View);
            
            return new VectorTensor(new GpuVectorStorage(result), [a, exponent], Backwards, NewGpuVectorStorage(Vectors.Zeros((int)result.Length)));

            void Backwards(VectorTensor output)
            {
                var gpuOutGrad = ToGpuVector(output.Gradient);
                var gpuOutVal = ToGpuVector(output.Value);
                var gpuAGrad = ToGpuVector(a.Gradient);
                var gpuExpGrad = ToGpuScalar(exponent.Gradient);
                var (
                    one, 
                    expMinusOne, 
                    scaled,
                    aPowExpMinusOne, 
                    logA,
                    mul1, mul2, sum) = (
                    AllocateTemp(1), 
                    AllocateTemp(1), 
                    AllocateTemp(a.Value.TotalElements),
                    AllocateTemp(a.Value.TotalElements), 
                    AllocateTemp(a.Value.TotalElements),
                    AllocateTemp(a.Value.TotalElements),
                    AllocateTemp(a.Value.TotalElements),
                    AllocateTemp(1));
                
                FillMemory(one, 1.0);
                SubtractMemory(gpuExponent, one, expMinusOne);
                PowerMemory(gpuA, expMinusOne, aPowExpMinusOne);
                ScaleMemory(aPowExpMinusOne, gpuExponent.GpuData, scaled);
                AccumulateInto(gpuAGrad, temp => MultiplyMemory(scaled, gpuOutGrad, temp));
                
                LogMemory(gpuA, logA);
                MultiplyMemory(gpuOutVal, logA, mul1);
                MultiplyMemory(gpuOutGrad, mul1, mul2);
                SumMemory(mul2, sum);
                AccumulateInto(gpuExpGrad, temp => CopyMemory(sum, temp));
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
    #endregion
}