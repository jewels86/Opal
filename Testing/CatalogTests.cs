using Opal.Autograd.Catalogs;
using Opal.Mathematics;
using static Opal.Autograd.Operations;

namespace Testing;

public static class CatalogTests
{
    private static void Assert(bool condition, string message)
    {
        if (!condition)
            throw new Exception($"Assertion failed: {message}");
    }
    
    private static bool ApproxEqual(double a, double b, double eps = 1e-6) => 
        Math.Abs(a - b) < eps;
    
    private static bool VectorApproxEqual(double[] a, double[] b, double eps = 1e-6)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (!ApproxEqual(a[i], b[i], eps)) return false;
        return true;
    }
    
    private static bool MatrixApproxEqual(double[,] a, double[,] b, double eps = 1e-6)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1)) 
            return false;
        for (int i = 0; i < a.GetLength(0); i++)
            for (int j = 0; j < a.GetLength(1); j++)
                if (!ApproxEqual(a[i,j], b[i,j], eps)) return false;
        return true;
    }

    public static void TestScalarCatalog()
    {
        Console.WriteLine("Testing ScalarCatalog...\n");
        var catalog = new ScalarCatalog();
        
        // Test 1: Vector * Scalar -> Sum (this is Multiply in catalog)
        Console.WriteLine("Test 1: Multiply (vector * scalar then sum)");
        var weights = NewVector([2.0, 3.0, 4.0]);
        var input = NewScalar(5.0, 0.0);
        
        var result = catalog.Multiply(weights, input);
        var resultValue = result.Value.ToHost();
        Console.WriteLine($"  Forward: {resultValue} (expected {2*5 + 3*5 + 4*5} = 45)");
        Assert(ApproxEqual(resultValue, 45.0), "Forward pass failed");
        
        result.Backward(NewDefaultScalarStorage(1.0));
        var weightsGrad = weights.Gradient.ToHost();
        var inputGrad = input.Gradient.ToHost();
        
        Console.WriteLine($"  Weights gradient: [{string.Join(", ", weightsGrad)}]");
        Console.WriteLine($"  Expected: [5.0, 5.0, 5.0]");
        Console.WriteLine($"  Input gradient: {inputGrad} (expected {2+3+4} = 9)");
        
        Assert(VectorApproxEqual(weightsGrad, [5.0, 5.0, 5.0]), "Weights gradient failed");
        Assert(ApproxEqual(inputGrad, 9.0), "Input gradient failed");
        Console.WriteLine("  ✓ Passed\n");
        
        // Test 2: Scalar Add
        Console.WriteLine("Test 2: Add (scalar + scalar)");
        var a = NewScalar(3.0, 0.0);
        var b = NewScalar(7.0, 0.0);
        
        var sum = catalog.Add(a, b);
        Console.WriteLine($"  Forward: {sum.Value.ToHost()} (expected 10.0)");
        Assert(ApproxEqual(sum.Value.ToHost(), 10.0), "Add forward failed");
        
        sum.Backward(NewCpuScalarStorage(1.0));
        Console.WriteLine($"  da: {a.Gradient.ToHost()} (expected 1.0)");
        Console.WriteLine($"  db: {b.Gradient.ToHost()} (expected 1.0)");
        Assert(ApproxEqual(a.Gradient.ToHost(), 1.0), "Add gradient a failed");
        Assert(ApproxEqual(b.Gradient.ToHost(), 1.0), "Add gradient b failed");
        Console.WriteLine("  ✓ Passed\n");
        
        // Test 3: Vector Subtract (storage operation)
        Console.WriteLine("Test 3: Subtract (vector storage)");
        var vecA = NewDefaultVectorStorage([10.0, 20.0, 30.0]);
        var vecB = NewDefaultVectorStorage([1.0, 2.0, 3.0]);
        var diff = catalog.Subtract(vecA, vecB);
        var diffHost = diff.ToHost();
        Console.WriteLine($"  Result: [{string.Join(", ", diffHost)}]");
        Console.WriteLine($"  Expected: [9.0, 18.0, 27.0]");
        Assert(VectorApproxEqual(diffHost, [9.0, 18.0, 27.0]), "Vector subtract failed");
        Console.WriteLine("  ✓ Passed\n");
        
        // Test 4: Vector Scale (storage operation)
        Console.WriteLine("Test 4: Scale (vector storage)");
        var vec = NewDefaultVectorStorage([2.0, 4.0, 6.0]);
        var scaled = catalog.Scale(vec, 0.5);
        var scaledHost = scaled.ToHost();
        Console.WriteLine($"  Result: [{string.Join(", ", scaledHost)}]");
        Console.WriteLine($"  Expected: [1.0, 2.0, 3.0]");
        Assert(VectorApproxEqual(scaledHost, [1.0, 2.0, 3.0]), "Vector scale failed");
        Console.WriteLine("  ✓ Passed\n");
        
        // Test 5: Parameter update simulation
        Console.WriteLine("Test 5: Parameter update simulation");
        var param = NewDefaultVectorStorage([1.0, 2.0, 3.0]);
        var grad = NewDefaultVectorStorage([0.1, 0.2, 0.3]);
        var lr = 0.1;
        
        var update = catalog.Scale(grad, lr);
        var newParam = catalog.Subtract(param, update);
        var newParamHost = newParam.ToHost();
        
        Console.WriteLine($"  Original: [{string.Join(", ", param.ToHost())}]");
        Console.WriteLine($"  Gradient: [{string.Join(", ", grad.ToHost())}]");
        Console.WriteLine($"  Updated:  [{string.Join(", ", newParamHost)}]");
        Console.WriteLine($"  Expected: [0.99, 1.98, 2.97]");
        Assert(VectorApproxEqual(newParamHost, [0.99, 1.98, 2.97]), "Parameter update failed");
        Console.WriteLine("  ✓ Passed\n");
        
        Console.WriteLine("✓ All ScalarCatalog tests passed!\n");
    }
    
    public static void TestVectorCatalog()
    {
        Console.WriteLine("Testing VectorCatalog...\n");
        var catalog = new VectorCatalog();
        
        // Test 1: Matrix * Vector
        Console.WriteLine("Test 1: Multiply (matrix * vector)");
        var matrix = NewMatrix(new[,] {
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        });
        var vector = NewVector([2.0, 3.0]);
        
        var result = catalog.Multiply(matrix, vector);
        var resultValue = result.Value.ToHost();
        // [1*2 + 2*3, 3*2 + 4*3, 5*2 + 6*3] = [8, 18, 28]
        Console.WriteLine($"  Forward: [{string.Join(", ", resultValue)}]");
        Console.WriteLine($"  Expected: [8.0, 18.0, 28.0]");
        Assert(VectorApproxEqual(resultValue, [8.0, 18.0, 28.0]), "Matrix-vector multiply forward failed");
        
        result.Backward(NewDefaultVectorStorage([1.0, 1.0, 1.0]));
        var matrixGrad = matrix.Gradient.ToHost();
        var vectorGrad = vector.Gradient.ToHost();
        
        Console.WriteLine($"  Matrix gradient:");
        for (int i = 0; i < matrixGrad.GetLength(0); i++)
            Console.WriteLine($"    [{matrixGrad[i,0]}, {matrixGrad[i,1]}]");
        Console.WriteLine($"  Expected: [[2, 3], [2, 3], [2, 3]]");
        
        Console.WriteLine($"  Vector gradient: [{string.Join(", ", vectorGrad)}]");
        Console.WriteLine($"  Expected: [9.0, 12.0]"); // [1+3+5, 2+4+6]
        
        Assert(MatrixApproxEqual(matrixGrad, new[,] {{2.0, 3.0}, {2.0, 3.0}, {2.0, 3.0}}), 
            "Matrix gradient failed");
        Assert(VectorApproxEqual(vectorGrad, [9.0, 12.0]), "Vector gradient failed");
        Console.WriteLine("  ✓ Passed\n");
        
        // Test 2: Vector Add
        Console.WriteLine("Test 2: Add (vector + vector)");
        var v1 = NewVector([1.0, 2.0, 3.0]);
        var v2 = NewVector([4.0, 5.0, 6.0]);
        
        var sum = catalog.Add(v1, v2);
        var sumValue = sum.Value.ToHost();
        Console.WriteLine($"  Forward: [{string.Join(", ", sumValue)}]");
        Console.WriteLine($"  Expected: [5.0, 7.0, 9.0]");
        Assert(VectorApproxEqual(sumValue, [5.0, 7.0, 9.0]), "Vector add forward failed");
        
        sum.Backward(NewDefaultVectorStorage([1.0, 1.0, 1.0]));
        Console.WriteLine($"  v1 gradient: [{string.Join(", ", v1.Gradient.ToHost())}]");
        Console.WriteLine($"  v2 gradient: [{string.Join(", ", v2.Gradient.ToHost())}]");
        Console.WriteLine($"  Expected: [1, 1, 1] for both");
        Assert(VectorApproxEqual(v1.Gradient.ToHost(), [1.0, 1.0, 1.0]), "v1 gradient failed");
        Assert(VectorApproxEqual(v2.Gradient.ToHost(), [1.0, 1.0, 1.0]), "v2 gradient failed");
        Console.WriteLine("  ✓ Passed\n");
        
        // Test 3: Matrix Subtract (storage)
        Console.WriteLine("Test 3: Subtract (matrix storage)");
        var matA = NewDefaultMatrixStorage(new[,] {{5.0, 6.0}, {7.0, 8.0}});
        var matB = NewDefaultMatrixStorage(new[,] {{1.0, 2.0}, {3.0, 4.0}});
        var matDiff = catalog.Subtract(matA, matB);
        var matDiffHost = matDiff.ToHost();
        Console.WriteLine($"  Result: [[{matDiffHost[0,0]}, {matDiffHost[0,1]}], [{matDiffHost[1,0]}, {matDiffHost[1,1]}]]");
        Console.WriteLine($"  Expected: [[4, 4], [4, 4]]");
        Assert(MatrixApproxEqual(matDiffHost, new[,] {{4.0, 4.0}, {4.0, 4.0}}), "Matrix subtract failed");
        Console.WriteLine("  ✓ Passed\n");
        
        // Test 4: Matrix Scale (storage)
        Console.WriteLine("Test 4: Scale (matrix storage)");
        var mat = NewDefaultMatrixStorage(new[,] {{2.0, 4.0}, {6.0, 8.0}});
        var scaledMat = catalog.Scale(mat, 0.5);
        var scaledMatHost = scaledMat.ToHost();
        Console.WriteLine($"  Result: [[{scaledMatHost[0,0]}, {scaledMatHost[0,1]}], [{scaledMatHost[1,0]}, {scaledMatHost[1,1]}]]");
        Console.WriteLine($"  Expected: [[1, 2], [3, 4]]");
        Assert(MatrixApproxEqual(scaledMatHost, new[,] {{1.0, 2.0}, {3.0, 4.0}}), "Matrix scale failed");
        Console.WriteLine("  ✓ Passed\n");
        
        Console.WriteLine("✓ All VectorCatalog tests passed!\n");
    }
    
    public static void RunAll()
    {
        Console.WriteLine("=== Running Catalog Tests ===\n");
        TestScalarCatalog();
        TestVectorCatalog();
        Console.WriteLine("=== All Catalog Tests Passed! ===\n");
    }
}