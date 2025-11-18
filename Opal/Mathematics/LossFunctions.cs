using static Opal.Autograd.Operations;

namespace Opal.Mathematics;

public static class LossFunctions
{
    #region Scalars
    public static ScalarTensor MeanSquaredError(ScalarTensor predicted, ScalarTensorStorage actual)
    {
        var actualTensor = NewScalar(actual, null, _ => { }, NewDefaultScalarStorage(0.0));
        var diff = Subtract(predicted, actualTensor);
        return Multiply(diff, diff);
    }
    
    #endregion
    #region Vectors
    public static ScalarTensor MeanSquaredErrorVector(VectorTensor predicted, VectorTensorStorage actual)
    {
        if (predicted.Value.TotalElements != actual.TotalElements)
            throw new ArgumentException("Vectors must be of the same length.");
        
        var actualTensor = NewVector(actual, null, _ => { }, NewDefaultVectorStorage(Vectors.Zeros(actual.TotalElements)));
        var diff = Subtract(predicted, actualTensor);
        var squared = Multiply(diff, diff);
        var sumSquared = Sum(squared);
        return Multiply(sumSquared, NewScalar(1.0 / actual.TotalElements, 0.0));
    }
    
    public static ScalarTensor MeanSquaredErrorVector(VectorTensor predicted, double[] actual)
    {
        if (predicted.Value.TotalElements != actual.Length)
            throw new ArgumentException("Vectors must be of the same length.");
        
        return MeanSquaredErrorVector(predicted, NewDefaultVectorStorage(actual));
    }
    
    public static ScalarTensor CrossEntropy(VectorTensor predicted, VectorTensorStorage actual)
    {
        if (predicted.Value.TotalElements != actual.TotalElements)
            throw new ArgumentException("Vectors must be of the same length.");

        var actualTensor = NewVector(actual, null, _ => { }, NewDefaultVectorStorage(Vectors.Zeros(actual.TotalElements)));
    
        var epsilon = Fill(predicted.Value.TotalElements, 1e-8, 0.0);
        var clampedPred = Add(predicted, epsilon);
    
        var logPred = Log(clampedPred);
        var product = Multiply(actualTensor, logPred);
        var sumProduct = Sum(product);
        var negSum = Negate(sumProduct);
        return Multiply(negSum, NewScalar(1.0 / actual.TotalElements, 0.0));
    }

    
    public static ScalarTensor CrossEntropy(VectorTensor predicted, double[] actual)
    {
        if (predicted.Value.TotalElements != actual.Length)
            throw new ArgumentException("Vectors must be of the same length.");
    
        return CrossEntropy(predicted, NewDefaultVectorStorage(actual));
    }

    public static ScalarTensor BinaryCrossEntropy(VectorTensor predicted, VectorTensorStorage actual)
    {
        if (predicted.Value.TotalElements != actual.TotalElements)
            throw new ArgumentException("Vectors must be of the same length.");

        var actualTensor = NewVector(actual, null, _ => { }, NewDefaultVectorStorage(Vectors.Zeros(actual.TotalElements)));
        var ones = Fill(predicted.Value.TotalElements, 1.0, 0.0);
        var epsilon = Fill(predicted.Value.TotalElements, 1e-8, 0.0);

        var clampedPred = Add(predicted, epsilon);
        var logPred = Log(clampedPred);
        var term1 = Multiply(actualTensor, logPred);

        var oneMinusActual = Subtract(ones, actualTensor);
        var oneMinusPred = Subtract(ones, clampedPred);
        var logOneMinusPred = Log(oneMinusPred);
        var term2 = Multiply(oneMinusActual, logOneMinusPred);

        var loss = Add(term1, term2);
        var sumLoss = Sum(Negate(loss));
        return Multiply(sumLoss, NewScalar(1.0 / actual.TotalElements, 0.0));
    }
    
    public static ScalarTensor BinaryCrossEntropy(VectorTensor predicted, double[] actual)
    {
        if (predicted.Value.TotalElements != actual.Length)
            throw new ArgumentException("Vectors must be of the same length.");
    
        return BinaryCrossEntropy(predicted, NewDefaultVectorStorage(actual));
    }
    #endregion
}

