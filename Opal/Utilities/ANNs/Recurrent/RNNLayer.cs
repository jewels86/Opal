namespace Opal.Utilities.ANNs.Recurrent;

public class RNNLayer : IRecurrentLayer
{
    public double[] State { get; private set; }
    public double[][] Weights { get; private set; }
    public ILayer.ActivationFunction ActivationFunction { get; set; }
    public ILayer.ActivationFunctionDerivative ActivationFunctionDerivative { get; set; }
    
    
    
    public RNNLayer(int inputSize, ILayer.ActivationFunction activationFunction, 
        ILayer.ActivationFunctionDerivative activationFunctionDerivative)
    {
        ActivationFunction = activationFunction;
        ActivationFunctionDerivative = activationFunctionDerivative;
        
    }

    public double[] Forward(double[] input)
    {
        throw new NotImplementedException();
    }
    public double[] Backward(double[] gradOutput, double learningRate)
    {
        throw new NotImplementedException();
    }

    public void Reset()
    {
        throw new NotImplementedException();
    }

    public double[][] Forward(double[] input, double[][] previousState)
    {
        throw new NotImplementedException();
    }

    public double[][] Backward(double[][] gradOutput, double learningRate)
    {
        throw new NotImplementedException();
    }
    
    public void ResetState()
    {
        
    }
}