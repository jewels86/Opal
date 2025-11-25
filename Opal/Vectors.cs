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
    
    #region Vector Operations
    
    #endregion
}