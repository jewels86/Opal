using Jewels.Lazulite;

namespace Jewels.Opal;

public partial class Operations
{
    public static Tensor<float> New(float scalar, float? gradient = null, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null, int? aidx = null) =>
        new(new ScalarValue(scalar, aidx ?? DefaultAcceleratorIndex), new ScalarValue(gradient ?? 0, aidx ?? DefaultAcceleratorIndex), backwardAction, inputs);
    public static Tensor<float> New(Value<float> scalar, Value<float> gradient, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null) 
        => new(scalar, gradient, backwardAction, inputs);

    public static Value<float> NewValue(float scalar) => new ScalarValue(scalar, DefaultAcceleratorIndex);
}