using ILGPU;
using ILGPU.Runtime;
using Opal.Mathematics;

namespace Opal.Autograd;

public static partial class Operations
{
    #region Matrix Tensor Helpers
    public static MatrixTensorStorage NewCpuMatrixStorage(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        return new CpuStorage<double[,]>(matrix, [rows, cols], rows * cols);
    }
    public static MatrixTensorStorage NewGpuMatrixStorage(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        var buffer = Operations.Accelerator.Allocate2DDenseX<double>(new Index2D(rows, cols));
        buffer.CopyFromCPU(matrix);
        return new GpuMatrixStorage(buffer);
    }
    public static MatrixTensorStorage NewDefaultMatrixStorage(double[,] matrix) => GpuAvailable ? NewGpuMatrixStorage(matrix) : NewCpuMatrixStorage(matrix);
    
    public static MatrixTensor NewMatrix(MatrixTensorStorage storage, List<object>? inputs, Action<Tensor<MatrixTensorStorage>> backwards,
        MatrixTensorStorage gradient) => new(storage, inputs, backwards, gradient);
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
    public static bool UseGpu(params MatrixTensorStorage[] storages) => storages.Any(s => s is GpuMatrixStorage) && GpuAvailable;
    public static GpuMatrixStorage ToGpuMatrix(MatrixTensorStorage storage) => storage as GpuMatrixStorage ?? (GpuMatrixStorage)storage.ToGpu();
    public static MemoryBuffer2D<double, Stride2D.DenseX> AllocateBuffer(in LongIndex2D extent) => Accelerator.Allocate2DDenseX<double>(extent);
    public static MatrixTensor BinaryOp(
        MatrixTensor a,
        MatrixTensor b,
        Action<Index2D, ArrayView2D<double, Stride2D.DenseX>, 
            ArrayView2D<double, Stride2D.DenseX>, 
            ArrayView2D<double, Stride2D.DenseX>> gpuKernel,
        Func<double[,], double[,], double[,]> cpuFallback,
        Action<MatrixTensor, MatrixTensor, MatrixTensor> gradientFn)
    {
        if (UseGpu(a.Value, b.Value))
        {
            var gpuA = ToGpuMatrix(a.Value);
            var gpuB = ToGpuMatrix(b.Value);
    
            var resultBuffer = AllocateBuffer(gpuA.GpuData.Extent);
    
            Queue.Enqueue(() => gpuKernel(
                gpuA.GpuData.Extent.ToIntIndex(), 
                gpuA.GpuData.View, 
                gpuB.GpuData.View, 
                resultBuffer.View));
    
            var resultStorage = new GpuMatrixStorage(resultBuffer);
            var gradStorage = new GpuMatrixStorage(AllocateBuffer(resultBuffer.Extent));
    
            return new MatrixTensor(resultStorage, [a, b], 
                output => gradientFn(a, b, output), gradStorage);
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
        MatrixTensorStorage gradient,
        MatrixTensorStorage incomingGrad)
    {
        if (UseGpu(gradient, incomingGrad))
        {
            var gpuGrad = ToGpuMatrix(gradient);
            var gpuIncoming = ToGpuMatrix(incomingGrad);

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
    public static MatrixTensorStorage AddStorage(MatrixTensorStorage a, MatrixTensorStorage b)
    {
        if (!UseGpu(a, b)) return NewCpuMatrixStorage(Matrices.Add(a.ToHost(), b.ToHost()));
        var gpuA = ToGpuMatrix(a);
        var gpuB = ToGpuMatrix(b);
        var result = AllocateBuffer(gpuA.GpuData.Extent);
            
        Queue.Enqueue(() => MatrixAddKernel(
            gpuA.GpuData.IntExtent,
            gpuA.GpuData.View,
            gpuB.GpuData.View,
            result.View));
        return new GpuMatrixStorage(result);
    }

    public static MatrixTensorStorage SubtractStorage(MatrixTensorStorage a, MatrixTensorStorage b)
    {
        if (!UseGpu(a, b)) return NewCpuMatrixStorage(Matrices.Add(a.ToHost(), b.ToHost()));
        var gpuA = ToGpuMatrix(a);
        var gpuB = ToGpuMatrix(b);
        var result = AllocateBuffer(gpuA.GpuData.Extent);
            
        Queue.Enqueue(() => MatrixSubtractKernel(
            gpuA.GpuData.IntExtent,
            gpuA.GpuData.View,
            gpuB.GpuData.View,
            result.View));
        return new GpuMatrixStorage(result);
    }

    public static MatrixTensorStorage ScaleMatrixStorage(MatrixTensorStorage matrix, ScalarTensorStorage scalar)
    {
        if (!UseGpu(matrix)) return NewCpuMatrixStorage(Matrices.Multiply(matrix.ToHost(), scalar.ToHost()));
        var gpuMatrix = ToGpuMatrix(matrix);
        var gpuScalar = ToGpuScalar(scalar);
        var result = AllocateBuffer(gpuMatrix.GpuData.Extent);
        
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
            var gpuMatrix = ToGpuMatrix(matrix.Value);
            var gpuVector = ToGpuVector(vector.Value);
            
            int rows = (int)gpuMatrix.GpuData.Extent.Y;
            var resultBuffer = AllocateBuffer(rows);
            
            Queue.Enqueue(() => MatrixVectorMultiplyKernel(
                rows,
                gpuMatrix.GpuData.View,
                gpuVector.GpuData.View,
                resultBuffer.View));
            
            var resultStorage = new GpuVectorStorage(resultBuffer);
            var gradStorage = new GpuVectorStorage(AllocateBuffer(rows));
            
            return new VectorTensor(resultStorage, [matrix, vector], Backward, gradStorage);
            
            void Backward(VectorTensor output)
            {
                var gradVectorBuffer = AllocateBuffer(gpuMatrix.GpuData.Extent.X);
                Queue.Enqueue(() => MatrixTransposeVectorMultiplyKernel(
                    (int)gpuMatrix.GpuData.Extent.X,
                    gpuMatrix.GpuData.View,
                    ((GpuVectorStorage)output.Gradient).GpuData.View,
                    gradVectorBuffer.View));
                AccumulateGradient(vector.Gradient, new GpuVectorStorage(gradVectorBuffer));
                
                var gradMatrixBuffer = AllocateBuffer(gpuMatrix.GpuData.Extent);
                Queue.Enqueue(() => OuterProductKernel(
                    gpuMatrix.GpuData.Extent.ToIntIndex(),
                    ((GpuVectorStorage)output.Gradient).GpuData.View,
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
            
            void Backward(VectorTensor output)
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