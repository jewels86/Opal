using Opal.Autograd;
using static Opal.Autograd.Operations;

namespace Opal.Mathematics;

public static class ActivationFunctions
{
    #region Scalars
    public static ScalarTensor ReLu(ScalarTensor x)
    {
        var xVal = x.Value.ToHost();
        var result = Math.Max(0, xVal);
        return new ScalarTensor(
            NewCpuScalarStorage(result),
            [x],
            Backwards,
            NewCpuScalarStorage(0.0));
        
        void Backwards(ScalarTensor output)
        {
            var grad = xVal > 0 ? output.Gradient.ToHost() : 0;
            x.Gradient.CopyFrom(x.Gradient.ToHost() + grad);
        }
    }

    public static ScalarTensor Sigmoid(ScalarTensor x)
    {
        var xVal = x.Value.ToHost();
        var s = 1.0 / (1.0 + Math.Exp(-xVal));
        return new ScalarTensor(
            NewCpuScalarStorage(s),
            [x],
            Backwards,
            NewCpuScalarStorage(0.0));
        
        void Backwards(ScalarTensor output)
        {
            var sig = output.Value.ToHost(); 
            var grad = output.Gradient.ToHost() * sig * (1 - sig);
            x.Gradient.CopyFrom(x.Gradient.ToHost() + grad);
        }
    }

    public static ScalarTensor Tanh(ScalarTensor x)
    {
        var xVal = x.Value.ToHost();
        var result = Math.Tanh(xVal);
        return new ScalarTensor(
            NewCpuScalarStorage(result),
            [x],
            Backwards,
            NewCpuScalarStorage(0.0));
        
        void Backwards(ScalarTensor output)
        {
            var grad = output.Gradient.ToHost() * (1 - result * result);
            x.Gradient.CopyFrom(x.Gradient.ToHost() + grad);
        }
    }
    
    public static ScalarTensor Identity(ScalarTensor x)
    {
        return new ScalarTensor(
            x.Value,
            [x],
            Backwards,
            NewDefaultScalarStorage(0.0));
        
        void Backwards(ScalarTensor output) => AccumulateGradient(x.Gradient, output.Gradient);
    }
    #endregion
    #region Vectors
    public static VectorTensor ReLuVector(VectorTensor x) => Operations.ReLuVector(x);

    public static VectorTensor SigmoidVector(VectorTensor x)
    {
        var negX = Negate(x);
        var expNegX = Exp(negX);
        var onePlusExp = Add(Fill(x.Value.TotalElements, 1.0, 0.0), expNegX);
        var ones = Fill(x.Value.TotalElements, 1.0, 0.0);
        return Divide(ones, onePlusExp).Defer();
    }

    public static VectorTensor TanhVector(VectorTensor x) => Operations.Tanh(x);
    
    public static VectorTensor IdentityVector(VectorTensor x)
    {
        return new VectorTensor(
            x.Value,
            [x],
            output => AccumulateGradient(x.Gradient, output.Gradient),
            NewDefaultVectorStorage(Vectors.Zeros(x.Value.TotalElements)));
    }

    public static VectorTensor SoftmaxVector(VectorTensor x)
    {
        using var expX = Exp(x);
        using var sumExp = Sum(expX);
        using var ones = Fill(x.Value.TotalElements, 1.0, 0.0);
        using var sumVector = Multiply(ones, sumExp);
    
        return new VectorTensor(
            DivideStorage(expX.Value, sumVector.Value),
            [x],
            Backwards,
            NewDefaultVectorStorage(Vectors.Zeros(x.Value.TotalElements)));

        void Backwards(VectorTensor output)
        {
            var softmax = output.Value.ToHost();
            var grad = new double[softmax.Length];
            var outGrad = output.Gradient.ToHost();
            for (int i = 0; i < softmax.Length; i++)
            {
                for (int j = 0; j < softmax.Length; j++)
                {
                    var delta = i == j ? 1.0 : 0.0;
                    grad[i] += outGrad[j] * softmax[i] * (delta - softmax[j]);
                }
            }
            AccumulateGradient(x.Gradient, NewDefaultVectorStorage(grad));
        }
    }
    #endregion
}
