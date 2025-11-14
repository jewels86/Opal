using Opal.Autograd;
using Opal.Autograd.Catalogs;
using Opal.Mathematics;

namespace Opal.NNs.Recurrent;

public class ScalarRecurrentNetwork : RecurrentNetwork<double, double, double, double, double, double, double>
{
    public ScalarRecurrentNetwork(
        int inputSize,
        int hiddenSize,
        int outputSize,
        int numHiddenLayers,
        ActivationFunction<double> hiddenActivation,
        ActivationFunction<double> outputActivation,
        LossFunction<double> lossFunction,
        string name = "ScalarRecurrentNetwork")
        : base(
            CreateLayer(hiddenSize, hiddenActivation),
            CreateHiddenLayers(numHiddenLayers, hiddenSize, hiddenActivation),
            CreateLayer(outputSize, outputActivation),
            lossFunction,
            hiddenSize,
            hiddenActivation,
            name)
    {
    }

    protected override RecurrentLayer<double, double, double, double> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenActivation);

    private static RecurrentLayer<double, double, double, double> CreateLayer(
        int outputSize,
        ActivationFunction<double> activation)
    {
        var catalog = new ScalarCatalog();
        var random = new Random();
        
        var inputWeights = new Tensor<double>[outputSize];
        for (int i = 0; i < outputSize; i++) 
        {
            var weight = random.NextDouble() * 2 - 1;
            inputWeights[i] = new Tensor<double>(weight, null, _ => { }, 0.0);
        }
        
        var recurrentWeights = new Tensor<double>[outputSize];
        for (int i = 0; i < outputSize; i++) 
        {
            var weight = random.NextDouble() * 2 - 1;
            recurrentWeights[i] = new Tensor<double>(weight, null, _ => { }, 0.0);
        }
        
        Tensor<double> biases = new(0.0, null, _ => { }, 0.0);
        Tensor<double> state = new(0.0, null, _ => { }, 0.0);
    
        return new RecurrentLayer<double, double, double, double>
        {
            InputWeights = inputWeights,
            RecurrentWeights = recurrentWeights,
            Biases = biases,
            State = state,
            Activation = activation,
            Catalog = catalog
        };
    }

    private static List<RecurrentLayer<double, double, double, double>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        ActivationFunction<double> activation)
    {
        var layers = new List<RecurrentLayer<double, double, double, double>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, activation));
        return layers;
    }
}
