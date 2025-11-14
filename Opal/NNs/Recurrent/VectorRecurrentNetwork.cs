using Opal.Autograd;
using Opal.Autograd.Catalogs;
using Opal.Mathematics;

namespace Opal.NNs.Recurrent;

public class VectorRecurrentNetwork : RecurrentNetwork<double[], double[], double[], double[], double[], double[], double[]>
{
    public VectorRecurrentNetwork(
        int inputSize,
        int hiddenSize,
        int outputSize,
        int numHiddenLayers,
        ActivationFunction<double[]> hiddenActivation,
        ActivationFunction<double[]> outputActivation,
        LossFunction<double[]> lossFunction,
        string name = "VectorRecurrentNetwork")
        : base(
            CreateLayer(inputSize, hiddenSize, hiddenActivation),
            CreateHiddenLayers(numHiddenLayers, hiddenSize, hiddenActivation),
            CreateLayer(hiddenSize, outputSize, outputActivation),
            lossFunction,
            hiddenSize,
            hiddenActivation,
            name)
    {
    }

    protected override RecurrentLayer<double[], double[], double[], double[]> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenSize, HiddenActivation);

    private static RecurrentLayer<double[], double[], double[], double[]> CreateLayer(
        int inputSize,
        int outputSize,
        ActivationFunction<double[]> activation)
    {
        var catalog = new VectorCatalog();
        var random = new Random();
        
        // Create input weights
        var inputWeights = new Tensor<double[]>[outputSize];
        for (int i = 0; i < outputSize; i++) 
        {
            var weight = new double[inputSize];  
            for (int j = 0; j < inputSize; j++)
                weight[j] = random.NextDouble() * 2 - 1;
            inputWeights[i] = new Tensor<double[]>(weight, null, _ => { }, Vectors.Zeros(inputSize));
        }
        
        // Create recurrent weights
        var recurrentWeights = new Tensor<double[]>[outputSize];
        for (int i = 0; i < outputSize; i++) 
        {
            var weight = new double[outputSize];  
            for (int j = 0; j < outputSize; j++)
                weight[j] = random.NextDouble() * 2 - 1;
            recurrentWeights[i] = new Tensor<double[]>(weight, null, _ => { }, Vectors.Zeros(outputSize));
        }
        
        // Create biases
        Tensor<double[]> biases = new(Vectors.Zeros(outputSize), null, _ => { }, Vectors.Zeros(outputSize));
        
        // Create initial state
        Tensor<double[]> state = new(Vectors.Zeros(outputSize), null, _ => { }, Vectors.Zeros(outputSize));
    
        return new RecurrentLayer<double[], double[], double[], double[]>
        {
            InputWeights = inputWeights,
            RecurrentWeights = recurrentWeights,
            Biases = biases,
            State = state,
            Activation = activation,
            Catalog = catalog
        };
    }

    private static List<RecurrentLayer<double[], double[], double[], double[]>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        ActivationFunction<double[]> activation)
    {
        var layers = new List<RecurrentLayer<double[], double[], double[], double[]>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize, activation));
        return layers;
    }
}

