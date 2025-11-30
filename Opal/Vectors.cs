using Jewels.Lazulite;

namespace Opal;

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
    public static Value<float[,]> Stack(float[][] vectors) => Stack(vectors.Select(NewValue).ToArray());
    #endregion
}