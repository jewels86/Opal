using Opal.Autograd;
using Opal.Autograd.Catalogs;
using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Ff;

public class ScalarFfNetwork : FfNetwork<double, double, double, double, double, double>
{
    public ScalarFfNetwork(
        int inputSize,
        int hiddenSize,
        int numHiddenLayers,
        ActivationFunction<double> hiddenActivation,
        ActivationFunction<double> outputActivation,
        LossFunction<double> lossFunction,
        string name = "ScalarFfNetwork")
        : base(
            CreateLayer(inputSize, hiddenActivation),
            CreateHiddenLayers(numHiddenLayers, hiddenSize, hiddenActivation),
            CreateLayer(hiddenSize, outputActivation),
            lossFunction,
            hiddenSize,
            hiddenActivation,
            name)
    {
    }

    protected override FfLayer<double, double, double> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenActivation);
    
    private static FfLayer<double, double, double> CreateLayer(
        int inputSize, 
        ActivationFunction<double> activation)
    {
        var catalog = new ScalarCatalog();
        var weights = new Tensor<double>[inputSize];
        
        var random = new Random();
        for (int i = 0; i < inputSize; i++)
        {
            double weight = random.NextDouble() * 2 - 1;
            weights[i] = new Tensor<double>(weight, null, _ => { }, 0.0);
        }
        
        var biases = new Tensor<double>(0.0, null, _ => { }, 0.0);
        
        return new FfLayer<double, double, double>(weights, biases, activation, catalog);
    }

    private static List<FfLayer<double, double, double>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        ActivationFunction<double> activation)
    {
        var layers = new List<FfLayer<double, double, double>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, activation));
        return layers;
    }
}

