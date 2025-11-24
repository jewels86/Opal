using Opal;
using Opal.Autograd.Catalogs;
using Opal.Mathematics;
using static Opal.Autograd.Operations;

namespace Testing;

public static class ScalarMultiplyDiagnosticTest
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

    public static void TestScalarMultiplyDetailed()
    {
        Console.WriteLine("=== Detailed ScalarCatalog.Multiply Diagnostic Test ===\n");
        
        var catalog = new ScalarCatalog();
        
        // Step 1: Test basic tensor creation
        Console.WriteLine("Step 1: Testing tensor creation...");
        var weights = NewVector([2.0, 3.0, 4.0]);
        var scalar = NewScalar(5.0, 0.0);
        
        Console.WriteLine($"  Weights: [{string.Join(", ", weights.Value.ToHost())}]");
        Console.WriteLine($"  Scalar: {scalar.Value.ToHost()}");
        Console.WriteLine("  ✓ Tensor creation successful\n");
        
        // Step 2: Test direct Operations.Multiply(VectorTensor, ScalarTensor)
        Console.WriteLine("Step 2: Testing Operations.Multiply(VectorTensor, ScalarTensor)...");
        Tensor<ITensorStorage<double[]>> multiplyResult;
        try
        {
            multiplyResult = Operations.Multiply(weights, scalar);
            var multiplyValues = multiplyResult.Value.ToHost();
            Console.WriteLine($"  Multiply result: [{string.Join(", ", multiplyValues)}]");
            Console.WriteLine($"  Expected: [10.0, 15.0, 20.0]");
            
            if (VectorApproxEqual(multiplyValues, [10.0, 15.0, 20.0]))
            {
                Console.WriteLine("  ✓ Vector-scalar multiplication working correctly");
            }
            else
            {
                Console.WriteLine("  ✗ Vector-scalar multiplication FAILED - this is the issue!");
                Console.WriteLine($"  Actual values: [{string.Join(", ", multiplyValues)}]");
                
                // Additional debugging
                Console.WriteLine("  \nDebugging multiply operation:");
                Console.WriteLine($"    Vector storage type: {weights.Value.GetType().Name}");
                Console.WriteLine($"    Scalar storage type: {scalar.Value.GetType().Name}");
                Console.WriteLine($"    GPU available: {Operations.GpuAvailable}");
                
                return; // Early exit since this is the root cause
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ✗ Operations.Multiply threw exception: {ex.Message}");
            return;
        }
        Console.WriteLine();
        
        // Step 3: Test Operations.Sum on the multiply result
        Console.WriteLine("Step 3: Testing Operations.Sum...");
        Tensor<ITensorStorage<double>> sumResult;
        try
        {
            sumResult = Operations.Sum(multiplyResult);
            var sumValue = sumResult.Value.ToHost();
            Console.WriteLine($"  Sum result: {sumValue}");
            Console.WriteLine($"  Expected: 45.0");
            
            if (ApproxEqual(sumValue, 45.0))
            {
                Console.WriteLine("  ✓ Sum operation working correctly");
            }
            else
            {
                Console.WriteLine("  ✗ Sum operation FAILED");
                Console.WriteLine($"  This means the issue is in the sum, not multiply");
                return;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ✗ Operations.Sum threw exception: {ex.Message}");
            return;
        }
        Console.WriteLine();
        
        // Step 4: Test the full ScalarCatalog.Multiply method
        Console.WriteLine("Step 4: Testing ScalarCatalog.Multiply (full method)...");
        try
        {
            var catalogResult = catalog.Multiply(weights, scalar);
            var catalogValue = catalogResult.Value.ToHost();
            Console.WriteLine($"  Catalog result: {catalogValue}");
            Console.WriteLine($"  Expected: 45.0");
            
            if (ApproxEqual(catalogValue, 45.0))
            {
                Console.WriteLine("  ✓ ScalarCatalog.Multiply working correctly");
            }
            else
            {
                Console.WriteLine("  ✗ ScalarCatalog.Multiply FAILED");
                Console.WriteLine("  This shouldn't happen if steps 2 and 3 passed!");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ✗ ScalarCatalog.Multiply threw exception: {ex.Message}");
            return;
        }
        Console.WriteLine();
        
        // Step 5: Test gradient computation
        Console.WriteLine("Step 5: Testing gradient computation...");
        try
        {
            var weightsGrad = NewVector([2.0, 3.0, 4.0]);
            var scalarGrad = NewScalar(5.0, 0.0);
            var result = catalog.Multiply(weightsGrad, scalarGrad);
            
            result.Backward(NewDefaultScalarStorage(1.0));
            
            var wGrad = weightsGrad.Gradient.ToHost();
            var sGrad = scalarGrad.Gradient.ToHost();
            
            Console.WriteLine($"  Weights gradient: [{string.Join(", ", wGrad)}]");
            Console.WriteLine($"  Expected: [5.0, 5.0, 5.0]");
            Console.WriteLine($"  Scalar gradient: {sGrad}");
            Console.WriteLine($"  Expected: 9.0");
            
            if (VectorApproxEqual(wGrad, [5.0, 5.0, 5.0]) && ApproxEqual(sGrad, 9.0))
            {
                Console.WriteLine("  ✓ Gradient computation working correctly");
            }
            else
            {
                Console.WriteLine("  ✗ Gradient computation FAILED");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ✗ Gradient computation threw exception: {ex.Message}");
        }
        Console.WriteLine();
        
        Console.WriteLine("=== Test Summary ===");
        Console.WriteLine("If Steps 1-4 all pass but you're getting zeros elsewhere,");
        Console.WriteLine("the issue might be in tensor lifecycle or memory management.");
        Console.WriteLine("If Step 2 fails, the issue is in Operations.Multiply(VectorTensor, ScalarTensor).");
        Console.WriteLine("If Step 3 fails, the issue is in Operations.Sum().");
    }
    
    public static void TestEdgeCases()
    {
        Console.WriteLine("\n=== Testing Edge Cases ===\n");
        
        var catalog = new ScalarCatalog();
        
        // Test with zeros
        Console.WriteLine("Test 1: Multiply with zero scalar...");
        var weights1 = NewVector([1.0, 2.0, 3.0]);
        var zero = NewScalar(0.0, 0.0);
        var result1 = catalog.Multiply(weights1, zero);
        Console.WriteLine($"  Result: {result1.Value.ToHost()} (expected 0.0)");
        
        // Test with zero vector
        Console.WriteLine("Test 2: Multiply zero vector with scalar...");
        var zeroWeights = NewVector([0.0, 0.0, 0.0]);
        var scalar2 = NewScalar(5.0, 0.0);
        var result2 = catalog.Multiply(zeroWeights, scalar2);
        Console.WriteLine($"  Result: {result2.Value.ToHost()} (expected 0.0)");
        
        // Test with negative values
        Console.WriteLine("Test 3: Multiply with negative values...");
        var weights3 = NewVector([-1.0, 2.0, -3.0]);
        var scalar3 = NewScalar(-2.0, 0.0);
        var result3 = catalog.Multiply(weights3, scalar3);
        var expected3 = (-1.0 * -2.0) + (2.0 * -2.0) + (-3.0 * -2.0); // 2 + (-4) + 6 = 4
        Console.WriteLine($"  Result: {result3.Value.ToHost()} (expected {expected3})");
        
        // Test with single element vector
        Console.WriteLine("Test 4: Multiply single element vector...");
        var weights4 = NewVector([7.0]);
        var scalar4 = NewScalar(3.0, 0.0);
        var result4 = catalog.Multiply(weights4, scalar4);
        Console.WriteLine($"  Result: {result4.Value.ToHost()} (expected 21.0)");
        
        // Test with large vector
        Console.WriteLine("Test 5: Multiply large vector...");
        var largeWeights = NewVector(Enumerable.Range(1, 100).Select(i => (double)i).ToArray());
        var scalar5 = NewScalar(0.1, 0.0);
        var result5 = catalog.Multiply(largeWeights, scalar5);
        var expected5 = Enumerable.Range(1, 100).Sum() * 0.1; // Sum of 1..100 = 5050, * 0.1 = 505
        Console.WriteLine($"  Result: {result5.Value.ToHost()} (expected {expected5})");
    }
    
    public static void TestMemoryAndSync()
    {
        Console.WriteLine("\n=== Testing Memory and Synchronization ===\n");
        
        Console.WriteLine("Test 1: Force CPU execution...");
        bool originalGpuSetting = Operations.GpuAvailable;
        Operations.GpuAvailable = false;
        
        try
        {
            var catalog = new ScalarCatalog();
            var weights = NewVector([2.0, 3.0, 4.0]);
            var scalar = NewScalar(5.0, 0.0);
            var result = catalog.Multiply(weights, scalar);
            Console.WriteLine($"  CPU Result: {result.Value.ToHost()} (expected 45.0)");
        }
        finally
        {
            Operations.GpuAvailable = originalGpuSetting;
        }
        
        if (originalGpuSetting)
        {
            Console.WriteLine("Test 2: Force GPU execution with explicit sync...");
            Operations.GpuAvailable = true;
            
            var catalog = new ScalarCatalog();
            var weights = NewVector([2.0, 3.0, 4.0]);
            var scalar = NewScalar(5.0, 0.0);
            
            // Explicit sync before and after
            Operations.Sync();
            var result = catalog.Multiply(weights, scalar);
            Operations.Sync();
            
            Console.WriteLine($"  GPU Result: {result.Value.ToHost()} (expected 45.0)");
        }
        else
        {
            Console.WriteLine("Test 2: GPU not available, skipping GPU-specific test.");
        }
    }
    
    public static void RunAll()
    {
        Console.WriteLine("=== Running ScalarCatalog.Multiply Diagnostic Tests ===\n");
        
        TestScalarMultiplyDetailed();
        TestEdgeCases();
        TestMemoryAndSync();
        
        Console.WriteLine("\n=== Diagnostic Tests Complete ===");
    }
}