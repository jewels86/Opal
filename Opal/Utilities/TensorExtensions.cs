using Opal.Utilities.Concurrency;

namespace Opal.Utilities;

public static class TensorExtensions
{
    #region Arithmetic Operations
    public static Tensor Add(this Tensor a, Tensor b, bool parallel = true)
    {
        if (!SameShape(a, b))
            throw new ArgumentException("Tensors must have the same shape for addition.");
        return new Tensor(a.Data
            .AsParallel(parallel)
            .Zip(b.Data.AsParallel(parallel), (x, y) => x + y)
            .ToArray(), a.Shape);
    }
    public static Tensor Subtract(this Tensor a, Tensor b, bool parallel = true) =>
        a.Add(b.Negate(parallel), parallel);

    public static Tensor Multiply(this Tensor a, Tensor b, bool parallel = true)
    {
        if (SameShape(a, b))
            return new(a.Data.AsParallel(parallel)
                .Zip(b.Data.AsParallel(parallel), (x, y) => x * y).ToArray(), a.Shape);
        
        switch (a.Shape.Length, b.Shape.Length)
        {
            case (2, 2):
                return MatrixMultiply(a, b, parallel);
            case (1, 1):
                double dot = DotProduct(a, b, parallel);
                return new Tensor([dot], [1]);
            default:
                return GeneralContraction(a, b, parallel);
        }
    }
    
    public static Tensor Negate(this Tensor a, bool parallel = true) =>
        new(a.Data.AsParallel(parallel).Select(x => -x).ToArray(), a.Shape);
    #endregion
    #region Private Helpers
    private static bool SameShape(Tensor a, Tensor b) => a.Shape.SequenceEqual(b.Shape);
    private static Tensor MatrixMultiply(Tensor a, Tensor b, bool parallel)
    {
        if (a.Shape.Length != 2 || b.Shape.Length != 2)
            throw new ArgumentException("Both tensors must be 2D for matrix multiplication.");
        if (a.Shape[1] != b.Shape[0])
            throw new ArgumentException("Inner dimensions must match for matrix multiplication.");
        
        int m = a.Shape[0], n = a.Shape[1], p = b.Shape[1];

        var range = Enumerable.Range(0, m * p);
        var query = parallel ? range.AsParallel() : range.AsEnumerable();

        double[] resultData = query
            .Select(idx =>
            {
                int i = idx / p, j = idx % p;
                double sum = 0;
                for (int k = 0; k < n; k++)
                    sum += a[i, k] * b[k, j];
                return sum;
            })
            .ToArray();

        return new Tensor(resultData, [m, p]);
    }
    private static double DotProduct(Tensor a, Tensor b, bool parallel)
    {
        if (a.Shape.Length != 1 || b.Shape.Length != 1)
            throw new ArgumentException("Both tensors must be 1D for dot product.");
        if (a.Shape[0] != b.Shape[0])
            throw new ArgumentException("Vectors must be the same length for dot product.");
        
        double sum = a.Data.AsParallel(parallel)
            .Zip(b.Data.AsParallel(parallel), (x, y) => x * y)
            .Sum();
        return sum;
    }
    private static Tensor GeneralContraction(Tensor a, Tensor b, bool parallel)
    {
        int contractDim = a.Shape[^1];
        if (b.Shape.Length == 0 || a.Shape.Length == 0 || b.Shape[0] != contractDim)
            throw new ArgumentException("Contracted dimensions must match.");

        int[] outShape = a.Shape.Take(a.Shape.Length - 1)
            .Concat(b.Shape.Skip(1)).ToArray();
        int outSize = outShape.Aggregate(1, (x, y) => x * y);

        var range = Enumerable.Range(0, outSize);
        var query = parallel ? range.AsParallel() : range.AsEnumerable();

        double[] resultData = query.Select(flatIdx =>
        {
            int[] resIdx = new int[outShape.Length];
            int rem = flatIdx;
            for (int d = outShape.Length - 1; d >= 0; d--)
            {
                resIdx[d] = rem % outShape[d];
                rem /= outShape[d];
            }

            int[] aIdx = new int[a.Shape.Length];
            int[] bIdx = new int[b.Shape.Length];
            Array.Copy(resIdx, 0, aIdx, 0, a.Shape.Length - 1);
            Array.Copy(resIdx, a.Shape.Length - 1, bIdx, 1, b.Shape.Length - 1);

            double sum = 0;
            for (int k = 0; k < contractDim; k++)
            {
                aIdx[^1] = k;
                bIdx[0] = k;
                sum += a[aIdx] * b[bIdx];
            }
            return sum;
        }).ToArray();

        return new Tensor(resultData, outShape);
    }
    #endregion
}