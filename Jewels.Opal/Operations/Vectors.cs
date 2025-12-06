using Jewels.Lazulite;

namespace Jewels.Opal;

public static partial class Operations
{
    #region Quick Helpers
    public static float[] Fill(float value, int size) => new float[size].Select(_ => value).ToArray();
    #endregion
    
    public static Tensor<float[]> New(float[] vector, float[]? gradient = null, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null, int? aidx = null) => 
        new(new VectorValue(vector, aidx ?? DefaultAcceleratorIndex), new VectorValue(gradient ?? Fill(0.0f, vector.Length), aidx ?? DefaultAcceleratorIndex), backwardAction, inputs);
    public static Tensor<float[]> New(Value<float[]> vector, Value<float[]> gradient, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null) => 
        new(vector, gradient, backwardAction, inputs);

    public static Value<float[]> NewValue(float[] vector) => new VectorValue(vector, DefaultAcceleratorIndex);
    public static Value<float[]>[] NewValues(params float[][] vectors) => vectors.Select(NewValue).ToArray();
    public static Value<float[]>[] NewValuesFromSingles(params float[] vectors) => vectors.Select(x => new[] { x }).Select(NewValue).ToArray();
    
    #region Value Operations
    public static Value<float> Dot(Value<float[]> a, Value<float[]> b)
    {
        var result = Compute.Get(a.AcceleratorIndex, 1);
        Compute.Dot(a, b, result);
        return new ScalarValue(result);
    }
    #endregion
    
    #region Vector Operations
    public static Tensor<float> Dot(Tensor<float[]> a, Tensor<float[]> b)
    {
        return new(Dot(a.Value, b.Value), new ScalarValue(0, a.AcceleratorIndex), Backward, [a, b]);
        
        void Backward(ITensor t)
        {
            MulScalarAccumulate(b.Value, t.Gradient, a.Gradient);
            MulScalarAccumulate(a.Value, t.Gradient, b.Gradient);
        }
    }
    #endregion
    
    #region Other things
    public static Value<float[,]> Stack(Value<float[]>[] vectors)
    {
        var (n, features, aidx) = (vectors.Length, vectors[0].TotalSize, vectors[0].AcceleratorIndex);
        var result = new MatrixValue(Compute.Get(aidx, n * features), [n, features]);
    
        for (int i = 0; i < n; i++)
            Compute.Call(Compute.CopyKernels, vectors[i].Data, result.Data.View.SubView(i * features, features));

        return result;
    }
    public static Value<float[,]> Stack(params float[][] vectors) => Stack(vectors.Select(NewValue).ToArray());
    public static Value<float[,]> StackSingles(params float[] vectors) => Stack(NewValuesFromSingles(vectors));

    public static Value<float[,,]> Stack(Value<float[]>[][] vectors)
    {
        var (n, sequences, features, aidx) = (vectors.Length, vectors[0].Length, vectors[0][0].TotalSize, vectors[0][0].AcceleratorIndex);
        var result = new TensorValue3(Compute.Get(aidx, n * sequences * features), [n, sequences, features]);
    
        for (int i = 0; i < n; i++)
            for (int j = 0; j < sequences; j++)
                Compute.Call(Compute.CopyKernels, vectors[i][j].Data, result.Data.View.SubView(i * sequences * features + j * features, features));
        
        return result;
    }
    public static Value<float[,,]> Stack(params float[][][] vectors) => Stack(vectors.Select(x => x.Select(NewValue).ToArray()).ToArray());

    public static Tensor<float[,,]> From(Value<float[,,]> tensor) => new(tensor, tensor.Zeros());
    #endregion
}