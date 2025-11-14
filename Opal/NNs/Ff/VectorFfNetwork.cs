using Opal.Autograd;
using Opal.Autograd.Catalogs;
using Opal.Mathematics;

namespace Opal.NNs.Ff;

public class VectorFfNetwork : FfNetwork<double[], double[], double[], double[], double[], double[]>
{
    public VectorFfNetwork(
        int inputSize,
        int hiddenSize,
        int outputSize,
        int numHiddenLayers,
        ActivationFunction<double[]> hiddenActivation,
        ActivationFunction<double[]> outputActivation,
        LossFunction<double[]> lossFunction,
        string name = "VectorFfNetwork")
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

    protected override FfLayer<double[], double[], double[]> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenSize, HiddenActivation);

    private static FfLayer<double[], double[], double[]> CreateLayer(
        int inputSize,
        int outputSize,
        ActivationFunction<double[]> activation)
    {
        var catalog = new VectorCatalog();
        var weights = new Tensor<double[]>[outputSize];
    
        var random = new Random();
        for (int i = 0; i < outputSize; i++) 
        {
            var weight = new double[inputSize];  
            for (int j = 0; j < inputSize; j++)
                weight[j] = random.NextDouble() * 2 - 1;
            weights[i] = new Tensor<double[]>(weight, null, _ => { }, Vectors.Zeros(inputSize));
        }
    
        Tensor<double[]> biases = new(Vectors.Zeros(outputSize), null, _ => { }, Vectors.Zeros(outputSize));
    
        return new FfLayer<double[], double[], double[]>(weights, biases, activation, catalog);
    }

    private static List<FfLayer<double[], double[], double[]>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        ActivationFunction<double[]> activation)
    {
        var layers = new List<FfLayer<double[], double[], double[]>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize, activation));
        return layers;
    }
}