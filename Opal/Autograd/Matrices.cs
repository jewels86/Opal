using ILGPU;
using ILGPU.Runtime;
using Opal.Mathematics;

namespace Opal.Autograd;

public static partial class Operations
{
    #region Matrix Tensor Helpers
    public static ITensorStorage<double[,]> NewCpuMatrixStorage(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        return new CpuStorage<double[,]>(matrix, [rows, cols], rows * cols);
    }
    public static ITensorStorage<double[,]> NewGpuMatrixStorage(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        var buffer = Operations.Accelerator.Allocate2DDenseX<double>(new Index2D(rows, cols));
        buffer.CopyFromCPU(matrix);
        return new GpuMatrixStorage(buffer);
    }
    public static ITensorStorage<double[,]> NewDefaultMatrixStorage(double[,] matrix) => GpuAvailable ? NewGpuMatrixStorage(matrix) : NewCpuMatrixStorage(matrix);
    
    public static MatrixTensor NewMatrix(ITensorStorage<double[,]> storage, List<object>? inputs, Action<Tensor<ITensorStorage<double[,]>>> backwards,
        ITensorStorage<double[,]> gradient) => new(storage, inputs, backwards, gradient);
    public static MatrixTensor NewMatrix(double[,] matrix, double[,] gradient) =>
        NewMatrix(NewDefaultMatrixStorage(matrix), null, _ => { }, NewDefaultMatrixStorage(gradient));
    #endregion
    #region Kernels
    public static Action<Index1D, ArrayView2D<double, Stride2D.DenseX>, 
            ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>> MatrixVectorMultiplyKernel { get; private set; }
    public static Action<Index1D, ArrayView2D<double, Stride2D.DenseX>,
        ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>> MatrixTransposeVectorMultiplyKernel { get; private set; }
    public static Action<Index2D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>> OuterProductKernel { get; private set; }
    public static Action<Index2D, ArrayView2D<double, Stride2D.DenseX>, 
        ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>> MatrixAddKernel { get; private set; }
    public static Action<Index2D, ArrayView2D<double, Stride2D.DenseX>, 
        ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>> MatrixSubtractKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>,
        ArrayView2D<double, Stride2D.DenseX>, int> CopyVectorToRowKernel { get; private set; }
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, int> ScaleVectorByRowKernel { get; private set; }
    public static Action<Index2D, ArrayView2D<double, Stride2D.DenseX>, 
        ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>> MatrixScalarMultiplyKernel { get; private set; }
    #endregion
    #region Helpers
    public static MatrixTensor BinaryOp(
        MatrixTensor a,
        MatrixTensor b,
        Action<Index2D, ArrayView2D<double, Stride2D.DenseX>, 
            ArrayView2D<double, Stride2D.DenseX>, 
            ArrayView2D<double, Stride2D.DenseX>> gpuKernel,
        Func<double[,], double[,], double[,]> cpuFallback,
        Action<MatrixTensor, MatrixTensor, MatrixTensor> gradientFn)
    {
        if (GpuAvailable && 
            (a.Value is GpuMatrixStorage || b.Value is GpuMatrixStorage))
        {
            var gpuA = (a.Value as GpuMatrixStorage) ?? 
                       (GpuMatrixStorage)a.Value.ToGpu();
            var gpuB = (b.Value as GpuMatrixStorage) ?? 
                       (GpuMatrixStorage)b.Value.ToGpu();
    
            var resultBuffer = Accelerator.Allocate2DDenseX<double>(gpuA.GpuData.Extent);
    
            Queue.Enqueue(() => gpuKernel(
                gpuA.GpuData.Extent.ToIntIndex(), 
                gpuA.GpuData.View, 
                gpuB.GpuData.View, 
                resultBuffer.View));
    
            var resultStorage = new GpuMatrixStorage(resultBuffer);
            var gradStorage = new GpuMatrixStorage(
                Accelerator.Allocate2DDenseX<double>(resultBuffer.Extent));
    
            return new MatrixTensor(resultStorage, [a, b], 
                output => gradientFn(a, b, (MatrixTensor)output), gradStorage);
        }
    
        var result = cpuFallback(a.Value.ToHost(), b.Value.ToHost());
        return new MatrixTensor(
            NewCpuMatrixStorage(result),
            [a, b],
            output => gradientFn(a, b, (MatrixTensor)output),
            NewCpuMatrixStorage(
                new double[result.GetLength(0), result.GetLength(1)]));
    }
    
    public static void AccumulateGradient(
        ITensorStorage<double[,]> gradient,
        ITensorStorage<double[,]> incomingGrad)
    {
        if (GpuAvailable &&
            (gradient is GpuMatrixStorage || incomingGrad is GpuMatrixStorage))
        {
            var gpuGrad = (gradient as GpuMatrixStorage) ??
                          (GpuMatrixStorage)gradient.ToGpu();
            var gpuIncoming = (incomingGrad as GpuMatrixStorage) ??
                              (GpuMatrixStorage)incomingGrad.ToGpu();

            Queue.Enqueue(() => MatrixAddKernel(
                gpuGrad.GpuData.IntExtent,
                gpuGrad.GpuData.View,
                gpuIncoming.GpuData.View,
                gpuGrad.GpuData.View));
        }
        else
        {
            var gradData = gradient.ToHost();
            var incomingData = incomingGrad.ToHost();
            gradient.CopyFrom(Matrices.Add(gradData, incomingData));
        }
    }
    #endregion
    #region Storage Operations
    public static ITensorStorage<double[,]> AddStorage(ITensorStorage<double[,]> a, ITensorStorage<double[,]> b)
    {
        if (!GpuAvailable || (a is not Autograd.GpuMatrixStorage && b is not Autograd.GpuMatrixStorage)) return NewCpuMatrixStorage(Matrices.Add(a.ToHost(), b.ToHost()));
        var gpuA = (a as GpuMatrixStorage) ?? (GpuMatrixStorage)a.ToGpu();
        var gpuB = (b as GpuMatrixStorage) ?? (GpuMatrixStorage)b.ToGpu();
        var result = Accelerator.Allocate2DDenseX<double>(gpuA.GpuData.Extent);
            
        Queue.Enqueue(() => MatrixAddKernel(
            gpuA.GpuData.IntExtent,
            gpuA.GpuData.View,
            gpuB.GpuData.View,
            result.View));
        return new GpuMatrixStorage(result);
    }

    public static ITensorStorage<double[,]> SubtractStorage(ITensorStorage<double[,]> a, ITensorStorage<double[,]> b)
    {
        if (!GpuAvailable || (a is not Autograd.GpuMatrixStorage && b is not Autograd.GpuMatrixStorage)) return NewCpuMatrixStorage(Matrices.Add(a.ToHost(), b.ToHost()));
        var gpuA = (a as GpuMatrixStorage) ?? (GpuMatrixStorage)a.ToGpu();
        var gpuB = (b as GpuMatrixStorage) ?? (GpuMatrixStorage)b.ToGpu();
        var result = Accelerator.Allocate2DDenseX<double>(gpuA.GpuData.Extent);
            
        Queue.Enqueue(() => MatrixSubtractKernel(
            gpuA.GpuData.IntExtent,
            gpuA.GpuData.View,
            gpuB.GpuData.View,
            result.View));
        return new GpuMatrixStorage(result);
    }

    public static ITensorStorage<double[,]> ScaleMatrixStorage(ITensorStorage<double[,]> matrix, ITensorStorage<double> scalar)
    {
        if (!GpuAvailable || matrix is not GpuMatrixStorage) return NewCpuMatrixStorage(Matrices.Multiply(matrix.ToHost(), scalar.ToHost()));
        var gpuMatrix = (matrix as GpuMatrixStorage) ?? (GpuMatrixStorage)matrix.ToGpu();
        var gpuScalar = (scalar as GpuScalarStorage) ?? (GpuScalarStorage)scalar.ToGpu();
        var result = Accelerator.Allocate2DDenseX<double>(gpuMatrix.GpuData.Extent);
        
        Queue.Enqueue(() => MatrixScalarMultiplyKernel(
            gpuMatrix.GpuData.IntExtent,
            gpuMatrix.GpuData.View,
            gpuScalar.GpuData.View,
            result.View));
        return new GpuMatrixStorage(result);;
    }
    #endregion
    
    #region Operations
    public static VectorTensor Multiply(MatrixTensor matrix, VectorTensor vector)
    {
        if (GpuAvailable && 
            (matrix.Value is GpuMatrixStorage || vector.Value is GpuVectorStorage))
        {
            var gpuMatrix = (matrix.Value as GpuMatrixStorage) ?? 
                            (GpuMatrixStorage)matrix.Value.ToGpu();
            var gpuVector = (vector.Value as GpuVectorStorage) ?? 
                            (GpuVectorStorage)vector.Value.ToGpu();
            
            int rows = (int)gpuMatrix.GpuData.Extent.Y;
            var resultBuffer = Accelerator.Allocate1D<double>(rows);
            
            Queue.Enqueue(() => MatrixVectorMultiplyKernel(
                rows,
                gpuMatrix.GpuData.View,
                gpuVector.GpuData.View,
                resultBuffer.View));
            
            var resultStorage = new GpuVectorStorage(resultBuffer);
            var gradStorage = new GpuVectorStorage(Accelerator.Allocate1D<double>(rows));
            
            return new VectorTensor(resultStorage, [matrix, vector], Backward, gradStorage);
            
            void Backward(Tensor<ITensorStorage<double[]>> output)
            {
                var outputVec = (VectorTensor)output;
                
                var gradVectorBuffer = Accelerator.Allocate1D<double>((int)gpuMatrix.GpuData.Extent.X);
                Queue.Enqueue(() => MatrixTransposeVectorMultiplyKernel(
                    (int)gpuMatrix.GpuData.Extent.X,
                    gpuMatrix.GpuData.View,
                    ((GpuVectorStorage)outputVec.Gradient).GpuData.View,
                    gradVectorBuffer.View));
                AccumulateGradient(vector.Gradient, new GpuVectorStorage(gradVectorBuffer));
                
                var gradMatrixBuffer = Accelerator.Allocate2DDenseX<double>(gpuMatrix.GpuData.Extent);
                Queue.Enqueue(() => OuterProductKernel(
                    gpuMatrix.GpuData.Extent.ToIntIndex(),
                    ((GpuVectorStorage)outputVec.Gradient).GpuData.View,
                    gpuVector.GpuData.View,
                    gradMatrixBuffer.View));
                AccumulateGradient(matrix.Gradient, new GpuMatrixStorage(gradMatrixBuffer));
            }
        }
        else
        {
            var result = Matrices.Multiply(matrix.Value.ToHost(), vector.Value.ToHost());
            return new VectorTensor(
                NewCpuVectorStorage(result),
                [matrix, vector],
                Backward,
                NewCpuVectorStorage(new double[result.Length]));
            
            void Backward(Tensor<ITensorStorage<double[]>> output)
            {
                var outGrad = output.Gradient.ToHost();
                var matrixVal = matrix.Value.ToHost();
                var vectorVal = vector.Value.ToHost();
                
                var gradVector = Matrices.Multiply(Matrices.Transpose(matrixVal), outGrad);
                AccumulateGradient(vector.Gradient, NewCpuVectorStorage(gradVector));
                
                var gradMatrix = Matrices.OuterProduct(outGrad, vectorVal);
                AccumulateGradient(matrix.Gradient, NewCpuMatrixStorage(gradMatrix));
            }
        }
    }
    #endregion
}