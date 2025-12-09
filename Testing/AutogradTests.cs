using System.Diagnostics;
using Jewels.Lazulite;
using Jewels.Opal;

namespace Testing;

public static class AutogradTests
{
    public static void TestScalarGradients()
    {
        int aidx = Operations.DefaultAcceleratorIndex;
        Console.WriteLine("Testing Scalar Autograd...");
        Stopwatch sw = Stopwatch.StartNew();
        
        // Build a simple graph: f(x,y,z) = (x * y) + z
        using var x = Operations.New(2);
        using var y = Operations.New(3);
        using var z = Operations.New(4);
        
        using var xy = Operations.Multiply(x, y);
        using var result = Operations.Add(xy, z);
        Operations.Sync();
        sw.Stop();
        
        // Forward: (2 * 3) + 4 = 10
        var resultValue = result.Value.ToHost();
        
        Console.WriteLine($"Forward pass: {resultValue} (expected 10.0 - {sw.ElapsedMilliseconds}ms)");
        Assert(Math.Abs(resultValue - 10.0) < 1e-10, "Forward pass failed");
        
        // Backward
        sw.Restart();
        result.Backward(new ScalarValue(1, aidx));
        Operations.Sync();
        sw.Stop();
        
        // Gradients: df/dx = y = 3, df/dy = x = 2, df/dz = 1
        var xGrad = x.Gradient.ToHost();
        var yGrad = y.Gradient.ToHost();
        var zGrad = z.Gradient.ToHost();
        
        Console.WriteLine($"dx: {xGrad} (expected 3.0 - {sw.ElapsedMilliseconds}ms)");
        Console.WriteLine($"dy: {yGrad} (expected 2.0)");
        Console.WriteLine($"dz: {zGrad} (expected 1.0)");
        
        Assert(Math.Abs(xGrad - 3.0) < 1e-10, "x gradient failed");
        Assert(Math.Abs(yGrad - 2.0) < 1e-10, "y gradient failed");
        Assert(Math.Abs(zGrad - 1.0) < 1e-10, "z gradient failed");
        
        Console.WriteLine("✓ Scalar test passed!\n");
    }
    
    public static void TestVectorGradients()
    {
        Console.WriteLine("Testing Vector Autograd...");
        Stopwatch sw = Stopwatch.StartNew();
        
        // f(a,b) = dot(a, b) where a = [1,2], b = [3,4]
        using var a = Operations.New([1, 2]);
        using var b = Operations.New([3, 4]);
        
        using var result = Operations.Dot(a, b);
        Operations.Sync();
        sw.Stop();
        
        // Forward: 1*3 + 2*4 = 11
        var resultValue = result.Value.ToHost();
        Console.WriteLine($"Forward pass: {resultValue} (expected 11.0 - {sw.ElapsedMilliseconds}ms)");
        Assert(Math.Abs(resultValue - 11.0) < 1e-10, "Forward pass failed");
        
        // Backward
        sw.Restart();
        result.Backward(new ScalarValue(1, a.AcceleratorIndex));
        Operations.Sync();
        sw.Stop();
        
        // Gradients: df/da = b = [3,4], df/db = a = [1,2]
        var aGrad = a.Gradient.ToHost();
        var bGrad = b.Gradient.ToHost();
        
        Console.WriteLine($"da: [{aGrad[0]}, {aGrad[1]}] (expected [3.0, 4.0] - {sw.ElapsedMilliseconds}ms)");
        Console.WriteLine($"db: [{bGrad[0]}, {bGrad[1]}] (expected [1.0, 2.0])");
        
        Assert(Math.Abs(aGrad[0] - 3.0) < 1e-10, "a[0] gradient failed");
        Assert(Math.Abs(aGrad[1] - 4.0) < 1e-10, "a[1] gradient failed");
        Assert(Math.Abs(bGrad[0] - 1.0) < 1e-10, "b[0] gradient failed");
        Assert(Math.Abs(bGrad[1] - 2.0) < 1e-10, "b[1] gradient failed");
        
        Console.WriteLine("✓ Vector test passed!\n");
    }
    
    public static void TestComplexGraph()
    {
        Console.WriteLine("Testing Complex Graph...");
        Stopwatch sw = Stopwatch.StartNew();
        
        // f(x,y) = (x + y) * (x - y)
        // At x=5, y=3: f = 8 * 2 = 16
        // df/dx = (x-y) + (x+y) = 2x = 10
        // df/dy = -(x+y) + (x-y) = -2y = -6
        
        using var x = Operations.New(5);
        using var y = Operations.New(3);
        
        using var sum = Operations.Add(x, y);
        using var diff = Operations.Subtract(x, y);
        using var result = Operations.Multiply(sum, diff);
        Operations.Sync();
        sw.Stop();
        
        var resultValue = result.Value.ToHost();
        
        Console.WriteLine($"Forward pass: {resultValue} (expected 16.0 - {sw.ElapsedMilliseconds}ms)");
        Assert(Math.Abs(resultValue - 16.0) < 1e-10, "Forward pass failed");
        
        sw.Restart();
        result.Backward(new ScalarValue(1, x.AcceleratorIndex));
        Operations.Sync();
        sw.Stop();
        
        var xGrad = x.Gradient.ToHost();
        var yGrad = y.Gradient.ToHost();
        
        Console.WriteLine($"dx: {xGrad} (expected 10.0 - {sw.ElapsedMilliseconds}ms)");
        Console.WriteLine($"dy: {yGrad} (expected -6.0)");
        
        Assert(Math.Abs(xGrad - 10.0) < 1e-10, "x gradient failed");
        Assert(Math.Abs(yGrad - (-6.0)) < 1e-10, "y gradient failed");
        
        Console.WriteLine("✓ Complex graph test passed!\n");
    }
    
    public static void TestMatrixMultiplyBackward()
    {
        Console.WriteLine("Testing Matrix Multiply Backward...");
        Stopwatch sw = Stopwatch.StartNew();
        
        // C = A * B where A is [2,3], B is [3,2]
        // Result C will be [2,2]
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        // B = [[1, 2],
        //      [3, 4],
        //      [5, 6]]
        // C = [[22, 28],
        //      [49, 64]]
        //
        // If grad_C = [[1, 1],
        //              [1, 1]]
        // grad_A = grad_C * B^T = [[1,1],[1,1]] * [[1,3,5],[2,4,6]] = [[3,7,11],[3,7,11]]
        // grad_B = A^T * grad_C = [[1,4],[2,5],[3,6]] * [[1,1],[1,1]] = [[5,5],[7,7],[9,9]]
        
        using var a = Operations.New(new float[,] {{1, 2, 3}, {4, 5, 6}});
        using var b = Operations.New(new float[,] {{1, 2}, {3, 4}, {5, 6}});
        
        using var c = Operations.MatrixMultiply(a, b);
        Operations.Sync();
        sw.Stop();
        
        var cValue = c.Value.ToHost();
        Console.WriteLine($"Forward pass: [{cValue[0,0]}, {cValue[0,1]}, {cValue[1,0]}, {cValue[1,1]}]");
        Console.WriteLine($"Expected: [22, 28, 49, 64] - {sw.ElapsedMilliseconds}ms");
        
        Assert(Math.Abs(cValue[0,0] - 22) < 1e-5, "Forward [0,0] failed");
        Assert(Math.Abs(cValue[0,1] - 28) < 1e-5, "Forward [0,1] failed");
        Assert(Math.Abs(cValue[1,0] - 49) < 1e-5, "Forward [1,0] failed");
        Assert(Math.Abs(cValue[1,1] - 64) < 1e-5, "Forward [1,1] failed");
        
        sw.Restart();
        c.Backward(Operations.NewValue(new float[,] {{1, 1}, {1, 1}}));
        Operations.Sync();
        sw.Stop();
        
        var aGrad = a.Gradient.ToHost();
        var bGrad = b.Gradient.ToHost();
        
        Console.WriteLine($"grad_A shape: [{aGrad.GetLength(0)}, {aGrad.GetLength(1)}] (expected [2, 3])");
        Console.WriteLine($"grad_B shape: [{bGrad.GetLength(0)}, {bGrad.GetLength(1)}] (expected [3, 2])");
        
        Console.WriteLine($"grad_A: [{aGrad[0,0]}, {aGrad[0,1]}, {aGrad[0,2]}, {aGrad[1,0]}, {aGrad[1,1]}, {aGrad[1,2]}]");
        Console.WriteLine($"Expected: [3, 7, 11, 3, 7, 11] - {sw.ElapsedMilliseconds}ms");
        Console.WriteLine($"grad_B: [{bGrad[0,0]}, {bGrad[0,1]}, {bGrad[1,0]}, {bGrad[1,1]}, {bGrad[2,0]}, {bGrad[2,1]}]");
        Console.WriteLine($"Expected: [5, 5, 7, 7, 9, 9]");
        
        Assert(Math.Abs(aGrad[0,0] - 3) < 1e-5, "grad_A[0,0] failed");
        Assert(Math.Abs(aGrad[0,1] - 7) < 1e-5, "grad_A[0,1] failed");
        Assert(Math.Abs(aGrad[0,2] - 11) < 1e-5, "grad_A[0,2] failed");
        Assert(Math.Abs(aGrad[1,0] - 3) < 1e-5, "grad_A[1,0] failed");
        Assert(Math.Abs(aGrad[1,1] - 7) < 1e-5, "grad_A[1,1] failed");
        Assert(Math.Abs(aGrad[1,2] - 11) < 1e-5, "grad_A[1,2] failed");
        
        Assert(Math.Abs(bGrad[0,0] - 5) < 1e-5, "grad_B[0,0] failed");
        Assert(Math.Abs(bGrad[0,1] - 5) < 1e-5, "grad_B[0,1] failed");
        Assert(Math.Abs(bGrad[1,0] - 7) < 1e-5, "grad_B[1,0] failed");
        Assert(Math.Abs(bGrad[1,1] - 7) < 1e-5, "grad_B[1,1] failed");
        Assert(Math.Abs(bGrad[2,0] - 9) < 1e-5, "grad_B[2,0] failed");
        Assert(Math.Abs(bGrad[2,1] - 9) < 1e-5, "grad_B[2,1] failed");
        
        Console.WriteLine("✓ Matrix multiply backward test passed!\n");
    }
    
    public static void TestMatrixMultiplyBackwardTransposeB()
    {
        Console.WriteLine("Testing Matrix Multiply Backward (transposeB=true)...");
        Stopwatch sw = Stopwatch.StartNew();
        
        // Simulating a batched neural network layer:
        // input: [batch=2, in_features=3]
        // weights: [out_features=2, in_features=3]
        // C = input @ weights.T where input is [2,3], weights.T is [3,2]
        // Result C will be [2,2]
        //
        // input = [[1, 2, 3],
        //          [4, 5, 6]]
        // weights = [[1, 2, 3],
        //            [4, 5, 6]]
        // weights.T = [[1, 4],
        //              [2, 5],
        //              [3, 6]]
        // C = input @ weights.T = [[14, 32],
        //                          [32, 77]]
        //
        // If grad_C = [[1, 1],
        //              [1, 1]]
        // grad_input = grad_C @ weights = [[1,1],[1,1]] @ [[1,2,3],[4,5,6]] = [[5,7,9],[5,7,9]]
        // grad_weights = grad_C.T @ input = [[1,1],[1,1]] @ [[1,2,3],[4,5,6]] = [[5,7,9],[5,7,9]]
        
        using var input = Operations.New(new float[,] {{1, 2, 3}, {4, 5, 6}});
        using var weights = Operations.New(new float[,] {{1, 2, 3}, {4, 5, 6}});
        
        using var c = Operations.MatrixMultiply(input, weights, transposeB: true);
        Operations.Sync();
        sw.Stop();
        
        var cValue = c.Value.ToHost();
        Console.WriteLine($"Forward pass: [{cValue[0,0]}, {cValue[0,1]}, {cValue[1,0]}, {cValue[1,1]}]");
        Console.WriteLine($"Expected: [14, 32, 32, 77] - {sw.ElapsedMilliseconds}ms");
        
        Assert(Math.Abs(cValue[0,0] - 14) < 1e-5, "Forward [0,0] failed");
        Assert(Math.Abs(cValue[0,1] - 32) < 1e-5, "Forward [0,1] failed");
        Assert(Math.Abs(cValue[1,0] - 32) < 1e-5, "Forward [1,0] failed");
        Assert(Math.Abs(cValue[1,1] - 77) < 1e-5, "Forward [1,1] failed");
        
        sw.Restart();
        c.Backward(Operations.NewValue(new float[,] {{1, 1}, {1, 1}}));
        Operations.Sync();
        sw.Stop();
        
        var inputGrad = input.Gradient.ToHost();
        var weightsGrad = weights.Gradient.ToHost();
        
        Console.WriteLine($"grad_input shape: [{inputGrad.GetLength(0)}, {inputGrad.GetLength(1)}] (expected [2, 3])");
        Console.WriteLine($"grad_weights shape: [{weightsGrad.GetLength(0)}, {weightsGrad.GetLength(1)}] (expected [2, 3])");
        
        Console.WriteLine($"grad_input: [{inputGrad[0,0]}, {inputGrad[0,1]}, {inputGrad[0,2]}, {inputGrad[1,0]}, {inputGrad[1,1]}, {inputGrad[1,2]}]");
        Console.WriteLine($"Expected: [5, 7, 9, 5, 7, 9] - {sw.ElapsedMilliseconds}ms");
        Console.WriteLine($"grad_weights: [{weightsGrad[0,0]}, {weightsGrad[0,1]}, {weightsGrad[0,2]}, {weightsGrad[1,0]}, {weightsGrad[1,1]}, {weightsGrad[1,2]}]");
        Console.WriteLine($"Expected: [5, 7, 9, 5, 7, 9]");
        
        Assert(Math.Abs(inputGrad[0,0] - 5) < 1e-5, "grad_input[0,0] failed");
        Assert(Math.Abs(inputGrad[0,1] - 7) < 1e-5, "grad_input[0,1] failed");
        Assert(Math.Abs(inputGrad[0,2] - 9) < 1e-5, "grad_input[0,2] failed");
        Assert(Math.Abs(inputGrad[1,0] - 5) < 1e-5, "grad_input[1,0] failed");
        Assert(Math.Abs(inputGrad[1,1] - 7) < 1e-5, "grad_input[1,1] failed");
        Assert(Math.Abs(inputGrad[1,2] - 9) < 1e-5, "grad_input[1,2] failed");
        
        Assert(Math.Abs(weightsGrad[0,0] - 5) < 1e-5, "grad_weights[0,0] failed");
        Assert(Math.Abs(weightsGrad[0,1] - 7) < 1e-5, "grad_weights[0,1] failed");
        Assert(Math.Abs(weightsGrad[0,2] - 9) < 1e-5, "grad_weights[0,2] failed");
        Assert(Math.Abs(weightsGrad[1,0] - 5) < 1e-5, "grad_weights[1,0] failed");
        Assert(Math.Abs(weightsGrad[1,1] - 7) < 1e-5, "grad_weights[1,1] failed");
        Assert(Math.Abs(weightsGrad[1,2] - 9) < 1e-5, "grad_weights[1,2] failed");
        
        Console.WriteLine("✓ Matrix multiply backward (transposeB) test passed!\n");
    }

    public static void TestMSEBackward()
    {
        Console.WriteLine("Testing MSE Backward...");
        Stopwatch sw = Stopwatch.StartNew();
        
        // predictions = [2, 4, 6]
        // targets = [1, 3, 5]
        // MSE = ((2-1)^2 + (4-3)^2 + (6-5)^2) / 3 = (1 + 1 + 1) / 3 = 1.0
        // grad = 2 * (pred - target) / n = 2 * [1, 1, 1] / 3 = [0.6667, 0.6667, 0.6667]
        
        using var pred = Operations.New([2, 4, 6]);
        using var target = Operations.NewValue([1, 3, 5]);
        
        using var loss = LossFunctions.MeanSquaredError(pred, target);
        Operations.Sync();
        sw.Stop();
        
        var lossValue = loss.Value.ToHost();
        Console.WriteLine($"Forward pass: {lossValue} (expected 1.0 - {sw.ElapsedMilliseconds}ms)");
        Assert(Math.Abs(lossValue - 1.0) < 1e-5, "MSE forward failed");
        
        sw.Restart();
        loss.Backward(Operations.NewValue(1));
        Operations.Sync();
        sw.Stop();
        
        var predGrad = pred.Gradient.ToHost();
        Console.WriteLine($"grad_pred: [{predGrad[0]}, {predGrad[1]}, {predGrad[2]}]");
        Console.WriteLine($"Expected: [0.6667, 0.6667, 0.6667] - {sw.ElapsedMilliseconds}ms");
        
        Assert(Math.Abs(predGrad[0] - 0.6667) < 1e-3, "grad_pred[0] failed");
        Assert(Math.Abs(predGrad[1] - 0.6667) < 1e-3, "grad_pred[1] failed");
        Assert(Math.Abs(predGrad[2] - 0.6667) < 1e-3, "grad_pred[2] failed");
        
        Console.WriteLine("✓ MSE backward test passed!\n");
    }
    
    public static void TestMatrixVectorAddBackward()
    {
        Console.WriteLine("Testing Matrix-Vector Add Backward...");
        Stopwatch sw = Stopwatch.StartNew();
        
        // Matrix: [[1, 2], [3, 4]]
        // Vector: [10, 20]
        // Result: [[11, 22], [13, 24]]
        // Gradient flows back:
        // - matrix grad = upstream grad (elementwise)
        // - vector grad = sum of upstream grad across rows
        
        using var matrix = Operations.New(new float[,] { { 1, 2 }, { 3, 4 } });
        using var vector = Operations.New([10, 20]);
        
        using var result = Operations.Add(matrix, vector);
        Operations.Sync();
        sw.Stop();
        
        var resultValue = result.Value.ToHost();
        Console.WriteLine($"Forward pass: [[{resultValue[0,0]}, {resultValue[0,1]}], [{resultValue[1,0]}, {resultValue[1,1]}]]");
        Console.WriteLine($"Expected: [[11, 22], [13, 24]] - {sw.ElapsedMilliseconds}ms");
        Assert(Math.Abs(resultValue[0,0] - 11) < 1e-5, "result[0,0] failed");
        Assert(Math.Abs(resultValue[0,1] - 22) < 1e-5, "result[0,1] failed");
        Assert(Math.Abs(resultValue[1,0] - 13) < 1e-5, "result[1,0] failed");
        Assert(Math.Abs(resultValue[1,1] - 24) < 1e-5, "result[1,1] failed");
        
        sw.Restart();
        var upstreamGrad = new float[,] { { 1, 1 }, { 1, 1 } };
        Operations.Sync();
        result.Backward(Operations.NewValue(upstreamGrad));
        Operations.Sync();
        sw.Stop();
        
        var matrixGrad = matrix.Gradient.ToProxy().FlatData;
        var vectorGrad = vector.Gradient.ToHost();
        
        Console.WriteLine($"grad_matrix: [[{matrixGrad[0]}, {matrixGrad[1]}], [{matrixGrad[2]}, {matrixGrad[3]}]]");
        Console.WriteLine($"Expected: [[1, 1], [1, 1]]");
        Console.WriteLine($"grad_vector: [{vectorGrad[0]}, {vectorGrad[1]}]");
        Console.WriteLine($"Expected: [2, 2] (sum across rows) - {sw.ElapsedMilliseconds}ms");
        
        Assert(Math.Abs(matrixGrad[0] - 1) < 1e-5, "grad_matrix[0,0] failed");
        Assert(Math.Abs(matrixGrad[1] - 1) < 1e-5, "grad_matrix[0,1] failed");
        Assert(Math.Abs(vectorGrad[0] - 2) < 1e-5, "grad_vector[0] failed");
        Assert(Math.Abs(vectorGrad[1] - 2) < 1e-5, "grad_vector[1] failed");
        
        Console.WriteLine("✓ Matrix-vector add backward test passed!\n");
    }

    public static void TestTanhBackward()
    {
        Console.WriteLine("Testing Tanh backward...");
        Stopwatch sw = Stopwatch.StartNew();
        
        // tanh(0) = 0, tanh'(0) = 1 - 0^2 = 1
        // tanh(1) = around 0.76, tanh'(0) = 1 - 0.76^2 = around 0.42
        // tanh(20) = 1, tanh'(20) = 1 - 1^2 = 0
        
        using var zero = Operations.New(0);
        using var one = Operations.New(1);
        using var twenty = Operations.New(20);
        
        using var tanhZero = ActivationFunctions.Tanh(zero);
        using var tanhOne = ActivationFunctions.Tanh(one);
        using var tanhTwenty = ActivationFunctions.Tanh(twenty);
        
        Operations.Sync();
        sw.Stop();
        
        var tanhZeroValue = tanhZero.Value.ToHost();
        var tanhOneValue = tanhOne.Value.ToHost();
        var tanhTwentyValue = tanhTwenty.Value.ToHost();
        
        Console.WriteLine($"Forward pass: {tanhZeroValue} (expected 0 - {sw.ElapsedMilliseconds}ms)");
        Console.WriteLine($"Forward pass: {tanhOneValue} (expected around 0.76 - {sw.ElapsedMilliseconds}ms)");
        Console.WriteLine($"Forward pass: {tanhTwentyValue} (expected 1 - {sw.ElapsedMilliseconds}ms)");
        Assert(Math.Abs(tanhZeroValue - 0) < 1e-5, "Tanh forward failed");
        Assert(Math.Abs(tanhOneValue - 0.76) < 1e-2, "Tanh forward failed");
        Assert(Math.Abs(tanhTwentyValue - 1) < 1e-5, "Tanh forward failed");
        
        sw.Restart();
        tanhZero.Backward(Operations.NewValue(1));
        tanhOne.Backward(Operations.NewValue(1));
        tanhTwenty.Backward(Operations.NewValue(1));
        Operations.Sync();
        sw.Stop();
        
        var zeroGrad = zero.Gradient.ToHost();
        var oneGrad = one.Gradient.ToHost();
        var twentyGrad = twenty.Gradient.ToHost();
        
        Console.WriteLine($"grad_zero: {zeroGrad} (expected 1 - {sw.ElapsedMilliseconds}ms)");
        Console.WriteLine($"grad_one: {oneGrad} (expected around 0.42 - {sw.ElapsedMilliseconds}ms)");
        Console.WriteLine($"grad_twenty: {twentyGrad} (expected 0 - {sw.ElapsedMilliseconds}ms)");
        
        Assert(Math.Abs(zeroGrad - 1) < 1e-5f, "grad_zero failed");
        Assert(Math.Abs(oneGrad - 0.42) < 1e-2f, "grad_one failed");
        Assert(Math.Abs(twentyGrad - 0) < 1e-5f, "grad_twenty failed");
        
        Console.WriteLine("Tests passed.");
    }

    public static void TestLstmGraphs()
    {
        Console.WriteLine("Testing Lstm State backward...");
        Stopwatch sw = Stopwatch.StartNew();
        
        // forget = 2, oldState = 3, input = 4, cell = 5
        // lstm state = forget * oldState + input * cell = 2 * 3 + 4 * 5 = 26
        // d/dforget = oldState = 3
        // d/doldState = forget = 2
        // d/dinput = cell = 5
        // d/dcell = input = 4
        
        using var forget = Operations.New(2);
        using var oldState = Operations.New(3);
        using var input = Operations.New(4);
        using var cell = Operations.New(5);
        
        using var state = Operations.LstmState(forget, oldState, input, cell);
        
        Operations.Sync();
        sw.Stop();
        
        var stateValue = state.Value.ToHost();
        Console.WriteLine($"State forward pass: {stateValue} (expected 26 - {sw.ElapsedMilliseconds}ms)");
        Assert(Math.Abs(stateValue - 26) < 1e-5, "LSTM state forward failed");
        
        sw.Restart();
        state.Backward(Operations.NewValue(1));
        Operations.Sync();
        
        var forgetGrad = forget.Gradient.ToHost();
        var oldStateGrad = oldState.Gradient.ToHost();
        var inputGrad = input.Gradient.ToHost();
        var cellGrad = cell.Gradient.ToHost();
        
        Console.WriteLine($"grad_forget: {forgetGrad} (expected 3 - {sw.ElapsedMilliseconds}ms)");
        Console.WriteLine($"grad_oldState: {oldStateGrad} (expected 2 - {sw.ElapsedMilliseconds}ms)");
        Console.WriteLine($"grad_input: {inputGrad} (expected 5 - {sw.ElapsedMilliseconds}ms)");
        Console.WriteLine($"grad_cell: {cellGrad} (expected 4 - {sw.ElapsedMilliseconds}ms)");
        
        Assert(Math.Abs(forgetGrad - 3) < 1e-5, "forgetGrad failed");
        Assert(Math.Abs(oldStateGrad - 2) < 1e-5, "oldStateGrad failed");
        Assert(Math.Abs(inputGrad - 5) < 1e-5, "oldStateGrad failed");
        Assert(Math.Abs(cellGrad - 4) < 1e-5, "oldStateGrad failed");
        
        // output = 5, newState is our state we made = 32
        // lstm hidden: output * tanh(newState) = 5 * tanh(32) = 5 * 1 = 5
        // d/doutput = tanh(newState) = 1
        // d/dnewState
        
        sw.Restart();
        using var output = Operations.New(5);
        using var hidden = Operations.Multiply(output, ActivationFunctions.Tanh(state));
        Operations.Sync();
        sw.Stop();
        
        var hiddenValue = hidden.Value.ToHost();
        Console.WriteLine($"\nHidden forward pass: {hiddenValue} (expected 5 - {sw.ElapsedMilliseconds}ms)");
        Assert(Math.Abs(hiddenValue - 5) < 1e-5, "LSTM hidden forward failed");
        
        sw.Restart();
        hidden.Backward(Operations.NewValue(1));
        Operations.Sync();
        sw.Stop();
        
        var outputGrad = output.Gradient.ToHost();
        var newStateGrad = state.Gradient.ToHost();
        
        Console.WriteLine($"grad_output: {outputGrad} (expected 1 - {sw.ElapsedMilliseconds}ms)");
        Console.WriteLine($"grad_newState: {newStateGrad} (expected 1 - {sw.ElapsedMilliseconds}ms)");
        
        Assert(Math.Abs(outputGrad - 1) < 1e-5, "outputGrad failed");
        Assert(Math.Abs(newStateGrad - 1) < 1e-5, "newStateGrad failed");
        
        Console.WriteLine("Tests passed...");
    }
    
    private static void Assert(bool condition, string message)
    {
        if (!condition)
            throw new Exception($"Assertion failed: {message}");
    }
    
    public static void RunAll()
    {
        TestScalarGradients();
        TestVectorGradients();
        TestComplexGraph();
        TestMatrixMultiplyBackward();
        TestMatrixMultiplyBackwardTransposeB();
        TestMSEBackward();
        TestMatrixVectorAddBackward();
        TestTanhBackward();
        TestLstmGraphs();
        Console.WriteLine("All tests passed! ✓");
    }
}