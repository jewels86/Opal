namespace Opal.Mathematics;

public static class Matrices
{
    private static readonly Random Random = new();
    
    public static double[,] RandomMatrix(int rows, int cols, double min = -1, double max = 1)
    {
        double[,] matrix = new double[rows, cols];
        double m = max - min;
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = Random.NextDouble() * m + min;
        return matrix;
    }
    
    #region Apply
    public static double[,] ApplyElementwise(double[,] input, Func<double, double> func)
    {
        int rows = input.GetLength(0), cols = input.GetLength(1);
        var result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[i, j] = func(input[i, j]);
        return result;
    }
    public static double[,] ApplyElementwise(double[,] input, Func<double, int, int, double> func)
    {
        int rows = input.GetLength(0), cols = input.GetLength(1);
        var result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[i, j] = func(input[i, j], i, j);
        return result;
    }
    #endregion
    
    #region Simple Operations
    public static double[,] Add(double[,] a, double[,] b)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            throw new ArgumentException("Matrices must be of the same dimensions.");
        return ApplyElementwise(a, (x, i, j) => x + b[i, j]);
    }

    public static double[,] Add(params List<double[,]> matrices)
    {
        var result = matrices[0];
        for (int i = 1; i < matrices.Count; i++)
            result = Add(result, matrices[i]);
        return result;
    }
    public static double[,] Subtract(double[,] a, double[,] b)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            throw new ArgumentException("Matrices must be of the same dimensions.");
        return ApplyElementwise(a, (x, i, j) => x - b[i, j]);
    }
    public static double[,] Multiply(double[,] a, double[,] b) 
    {
        int aRows = a.GetLength(0), aCols = a.GetLength(1);
        int bRows = b.GetLength(0), bCols = b.GetLength(1);
        if (aCols != bRows)
            throw new ArgumentException("Matrix A columns must match Matrix B rows.");
        
        var result = new double[aRows, bCols];
        for (int i = 0; i < aRows; i++)
        for (int j = 0; j < bCols; j++)
        {
            double sum = 0;
            for (int k = 0; k < aCols; k++)
                sum += a[i, k] * b[k, j];
            result[i, j] = sum;
        }
        return result;
    }

    public static double[,] Multiply(params List<double[,]> matrices)
    {
        var result = matrices[0];
        for (int i = 1; i < matrices.Count; i++)
            result = Multiply(result, matrices[i]);
        return result;
    }
    #endregion
    
    #region Vectors and Matrices
    public static double[] Multiply(double[,] matrix, double[] vector)
    {
        int rows = matrix.GetLength(0), cols = matrix.GetLength(1);
        if (cols != vector.Length)
            throw new ArgumentException("Matrix columns must match vector size.");
        
        var result = new double[rows];
        for (int i = 0; i < rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < cols; j++)
                sum += matrix[i, j] * vector[j];
            result[i] = sum;
        }
        return result;
    }
    public static double[,] MultiplyMatrixByVectorAsColumn(double[,] matrix, double[] vector)
    {
        int m = matrix.GetLength(0);
        int n = matrix.GetLength(1);
        if (n != vector.Length)
            throw new ArgumentException("Matrix columns must match vector size.");
        var result = new double[m, 1];
        for (int i = 0; i < m; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++)
                sum += matrix[i, j] * vector[j];
            result[i, 0] = sum;
        }
        return result;
    }

    public static double[] MultiplyMatrixTransposeByColumn(double[,] matrix, double[,] column)
    {
        int m = matrix.GetLength(0);
        int n = matrix.GetLength(1);
        if (column.GetLength(0) != m || column.GetLength(1) != 1)
            throw new ArgumentException("Column vector shape must be (m x 1)");
        double[] result = new double[n];
        for (int j = 0; j < n; j++)
        {
            double sum = 0;
            for (int i = 0; i < m; i++)
                sum += matrix[i, j] * column[i, 0];
            result[j] = sum;
        }
        return result;
    }

    public static double[,] OuterProductColumnAndVector(double[,] column, double[] vector)
    {
        int m = column.GetLength(0);
        int n = vector.Length;
        if (column.GetLength(1) != 1)
            throw new ArgumentException("First argument must be a column vector (m x 1)");
        double[,] result = new double[m, n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result[i, j] = column[i, 0] * vector[j];
        return result;
    }

    public static double[,] Concat(double[,] a, double[,] b)
    {
        int aRows = a.GetLength(0), aCols = a.GetLength(1);
        int bRows = b.GetLength(0), bCols = b.GetLength(1);
        if (aRows != bRows)
            throw new ArgumentException("Matrices must have the same number of rows to concatenate.");
        
        var result = new double[aRows, aCols + bCols];
        for (int i = 0; i < aRows; i++)
        {
            for (int j = 0; j < aCols; j++)
                result[i, j] = a[i, j];
            for (int j = 0; j < bCols; j++)
                result[i, aCols + j] = b[i, j];
        }
        return result;
    }
    #endregion
    #region Scalars and Matrices
    public static double[,] Multiply(double[,] matrix, double scalar) => ApplyElementwise(matrix, x => x * scalar);
    public static double[,] Divide(double[,] matrix, double scalar) => ApplyElementwise(matrix, x => x / scalar);
    #endregion

    #region Other Operations
    public static double[,] Transpose(double[,] matrix)
    {
        int rows = matrix.GetLength(0), cols = matrix.GetLength(1);
        var result = new double[cols, rows];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[j, i] = matrix[i, j];
        return result;
    }
    
    public static double[] Flatten(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        double[] flat = new double[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                flat[i * cols + j] = matrix[i, j];
        return flat;
    }

    public static double[,] OuterProduct(double[] a, double[] b)
    {
        int m = a.Length;
        int n = b.Length;
        double[,] result = new double[m, n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result[i, j] = a[i] * b[j];
        return result;
    }

    public static double[,] ToColumnVector(double[] vector)
    {
        int n = vector.Length;
        var result = new double[n, 1];
        for (int i = 0; i < n; i++)
            result[i, 0] = vector[i];
        return result;
    }

    public static double[,] ToSquareMatrix(double[] vector)
    {
        int n = vector.Length;
        int dim = (int)Math.Sqrt(n);
        if (dim * dim != n) dim++;
        var result = new double[dim, dim];
        for (int i = 0; i < n; i++)
            result[i / dim, i % dim] = vector[i];
        return result;
    }
    public static double Dot(double[,] a, double[,] b)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            throw new ArgumentException("Matrices must be of the same dimensions.");
        double sum = 0;
        for (int i = 0; i < a.GetLength(0); i++)
        for (int j = 0; j < a.GetLength(1); j++)
            sum += a[i, j] * b[i, j];
        return sum;
    }
    #endregion
    
    public static double[,] Zeros(int rows, int cols) => new double[rows, cols];
}