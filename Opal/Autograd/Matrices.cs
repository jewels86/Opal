using ILGPU;
using ILGPU.Runtime;
using Opal.Autograd.Gpu;
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
        var buffer = Controller.Get(rows, cols);
        buffer.CopyFromCPU(matrix);
        return new GpuMatrixStorage(buffer);
    }
    public static MatrixTensorStorage NewDefaultMatrixStorage(double[,] matrix) => GpuAvailable ? NewGpuMatrixStorage(matrix) : NewCpuMatrixStorage(matrix);
    
    public static MatrixTensor NewMatrix(MatrixTensorStorage storage, List<ITensor>? inputs, Action<MatrixTensor> backwards,
        MatrixTensorStorage gradient) => new(storage, inputs, backwards, gradient);
    public static MatrixTensor NewMatrix(double[,] matrix, double[,] gradient) =>
        NewMatrix(NewDefaultMatrixStorage(matrix), null, _ => { }, NewDefaultMatrixStorage(gradient));
    public static MatrixTensor NewMatrix(double[,] matrix) => NewMatrix(matrix, new double[matrix.GetLength(0), matrix.GetLength(1)]);
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
    public static Action<Index2D, ArrayView2D<double, Stride2D.DenseX>, 
        ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>> MatrixScalarMultiplyKernel { get; private set; }
    public static Action<Index2D, ArrayView2D<double, Stride2D.DenseX>, double> MatrixFillKernel { get; private set; }
    public static Action<Index1D, ArrayView2D<double, Stride2D.DenseX>, 
        ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>> MatrixTransposeVectorMultiplyAccumulateKernel { get; private set; }
    public static Action<Index2D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>> OuterProductAccumulateKernel { get; private set; }
    public static Action<Index2D, ArrayView2D<double, Stride2D.DenseX>, 
        ArrayView2D<double, Stride2D.DenseX>> MatrixCopyKernel { get; private set; }
    #endregion
    #region Helpers
    public static bool UseGpu(params MatrixTensorStorage[] storages) => storages.Any(s => s is GpuMatrixStorage) && GpuAvailable;
    public static GpuMatrixStorage ToGpuMatrix(MatrixTensorStorage storage) => storage as GpuMatrixStorage ?? (GpuMatrixStorage)storage.ToGpu();
    public static MemoryBuffer2D<double, Stride2D.DenseX> AllocateBuffer(in LongIndex2D extent) => Controller.Get((int)extent.X, (int)extent.Y);
    public static MemoryBuffer2D<double, Stride2D.DenseX> AllocateTemp(in LongIndex2D extent) => Controller.GetTemp((int)extent.X, (int)extent.Y);
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
    
            gpuKernel(
                gpuA.GpuData.Extent.ToIntIndex(), 
                gpuA.GpuData.View, 
                gpuB.GpuData.View, 
                resultBuffer.View);
    
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
        if (!UseGpu(gradient, incomingGrad))
        {
            var gradData = gradient.ToHost();
            var incomingData = incomingGrad.ToHost();
            gradient.CopyFrom(Matrices.Add(gradData, incomingData));
            return;
        }

        var gpuGrad = ToGpuMatrix(gradient);
        var gpuIncoming = ToGpuMatrix(incomingGrad);

        MatrixAddKernel(
            gpuGrad.GpuData.IntExtent,
            gpuGrad.GpuData.View,
            gpuIncoming.GpuData.View,
            gpuGrad.GpuData.View);

        if (gradient is GpuMatrixStorage) return;
        var rows = (int)gpuGrad.GpuData.Extent.X;
        var cols = (int)gpuGrad.GpuData.Extent.Y;
        var result = new double[rows, cols];
        gpuGrad.GpuData.CopyToCPU(result);
        gradient.CopyFrom(result);

    }

    public static void AccumulateInto(
        ArrayView2D<double, Stride2D.DenseX> gradient,
        Action<ArrayView2D<double, Stride2D.DenseX>> computeIntoTemp,
        bool subtract = false)
    {
        var temp = AllocateTemp(gradient.IntExtent);
        computeIntoTemp(temp.View);
        if (!subtract) MatrixAddKernel(temp.IntExtent, gradient, temp.View, gradient);
        else MatrixSubtractKernel(temp.IntExtent, gradient, temp.View, gradient);
    }
    #endregion
    #region Storage Operations
    public static MatrixTensorStorage AddStorage(MatrixTensorStorage a, MatrixTensorStorage b)
    {
        if (!UseGpu(a, b)) return NewCpuMatrixStorage(Matrices.Add(a.ToHost(), b.ToHost()));
        var gpuA = ToGpuMatrix(a);
        var gpuB = ToGpuMatrix(b);
        var result = AllocateBuffer(gpuA.GpuData.Extent);
            
        MatrixAddKernel(
            gpuA.GpuData.IntExtent,
            gpuA.GpuData.View,
            gpuB.GpuData.View,
            result.View);
        return new GpuMatrixStorage(result);
    }

    public static MatrixTensorStorage SubtractStorage(MatrixTensorStorage a, MatrixTensorStorage b)
    {
        if (!UseGpu(a, b)) return NewCpuMatrixStorage(Matrices.Subtract(a.ToHost(), b.ToHost()));
        var gpuA = ToGpuMatrix(a);
        var gpuB = ToGpuMatrix(b);
        var result = AllocateBuffer(gpuA.GpuData.Extent);
            
        MatrixSubtractKernel(
            gpuA.GpuData.IntExtent,
            gpuA.GpuData.View,
            gpuB.GpuData.View,
            result.View);
        return new GpuMatrixStorage(result);
    }

    public static MatrixTensorStorage ScaleMatrixStorage(MatrixTensorStorage matrix, ScalarTensorStorage scalar)
    {
        if (!UseGpu(matrix)) return NewCpuMatrixStorage(Matrices.Multiply(matrix.ToHost(), scalar.ToHost()));
        var gpuMatrix = ToGpuMatrix(matrix);
        var gpuScalar = ToGpuScalar(scalar);
        var result = AllocateBuffer(gpuMatrix.GpuData.Extent);
        
        MatrixScalarMultiplyKernel(
            gpuMatrix.GpuData.IntExtent,
            gpuMatrix.GpuData.View,
            gpuScalar.GpuData.View,
            result.View);
        return new GpuMatrixStorage(result);
    }

    public static void FillStorage(MatrixTensorStorage matrix, double value)
    {
        if (!UseGpu(matrix)) ((CpuStorage<double[,]>)matrix).Data = Matrices.Fill(matrix.Shape[0], matrix.Shape[1], value);
        var gpuMatrix = ToGpuMatrix(matrix);
        MatrixFillKernel(gpuMatrix.GpuData.IntExtent, gpuMatrix.GpuData.View, value);
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
            
            int rows = (int)gpuMatrix.GpuData.Extent.X;
            var resultBuffer = AllocateBuffer(rows);
            
            MatrixVectorMultiplyKernel(
                rows,
                gpuMatrix.GpuData.View,
                gpuVector.GpuData.View,
                resultBuffer.View);
            
            var resultStorage = new GpuVectorStorage(resultBuffer);
            var gradStorage = new GpuVectorStorage(AllocateBuffer(rows));
            
            return new VectorTensor(resultStorage, [matrix, vector], Backward, gradStorage);
            
            void Backward(VectorTensor output)
            {
                var gpuVectorGrad = ToGpuVector(vector.Gradient);
                var gpuMatrixGrad = ToGpuMatrix(matrix.Gradient);
                MatrixTransposeVectorMultiplyAccumulateKernel(
                    (int)gpuMatrix.GpuData.Extent.Y,
                    gpuMatrix.GpuData.View,
                    ((GpuVectorStorage)output.Gradient).GpuData.View,
                    gpuVectorGrad.GpuData.View); 
                
                OuterProductAccumulateKernel(
                    gpuMatrix.GpuData.Extent.ToIntIndex(),
                    ((GpuVectorStorage)output.Gradient).GpuData.View,
                    gpuVector.GpuData.View,
                    gpuMatrixGrad.GpuData.View);
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