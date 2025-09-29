using Opal.Mathematics;

namespace Opal.NNs.Lstm.Attention;

public abstract class LstmAttentionLayer<TWeights, TBiases, TTensor> : LstmLayer<TWeights, TBiases, TTensor>
    where TWeights : notnull
    where TBiases : notnull
    where TTensor : notnull
{
    protected readonly List<double[]> AttentionScores = [];
    
    protected new readonly ILstmAttentionTensorOperations<TWeights, TBiases, TTensor> TensorOperations;
    
    public LstmAttentionLayer(int[] inputShape, int[] hiddenShape, int[] outputShape,
        ILstmAttentionTensorOperations<TWeights, TBiases, TTensor> tensorOperations, IOptimizer<TWeights, TBiases> optimizer,
        ActivationFunction<TTensor> sigmoidActivation, ActivationFunction<TTensor> tanhActivation)
        : base(inputShape, hiddenShape, outputShape, tensorOperations, optimizer, sigmoidActivation, tanhActivation)
    {
        TensorOperations = tensorOperations;
    }

    public abstract double[] Alignment(TTensor[] hidden, TTensor prevState);
    public abstract void TrainAlignment(int timeStep, double[] gradScores, double learningRate);

    public TTensor Attention(TTensor[] hidden, TTensor prevState)
    {
        double[] scores = Alignment(hidden, prevState);
        double[] weights = Tensors.Softmax(scores);
        TTensor context = TensorOperations.WeightedSum(hidden, weights);
        AttentionScores.Add(scores);
        return context;
    }

    public TTensor[] AttentionBackwards(TTensor[] gradOutputs, TTensor[] encoderHiddenStates, double learningRate)
    {
        int timeSteps = gradOutputs.Length;
        int encoderSteps = encoderHiddenStates.Length;
        TTensor[] gradEncoderHidden = new TTensor[encoderSteps];
        
        for (int i = 0; i < encoderSteps; i++)
            gradEncoderHidden[i] = TensorOperations.DefaultState(HiddenShape);

        for (int t = 0; t < timeSteps; t++)
        {
            double[] scores = AttentionScores[t];
            double[] weights = Tensors.Softmax(scores);
            
            TTensor gradContext = gradOutputs[t];

            for (int i = 0; i < encoderSteps; i++)
            {
                gradEncoderHidden[i] = TensorOperations.Add(
                    gradEncoderHidden[i],
                    TensorOperations.Multiply(gradContext, weights[i])
                );
            }
            
            double[] gradWeights = new double[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                TTensor encoderHidden = encoderHiddenStates[i];
                gradWeights[i] = TensorOperations.Dot(gradContext, encoderHidden);
            }
            
            double[] gradScores = new double[scores.Length];
            for (int i = 0; i < scores.Length; i++)
            {
                double sum = 0;
                for (int j = 0; j < scores.Length; j++)
                {
                    double delta = (i == j) ? 1 : 0;
                    sum += gradWeights[j] * weights[j] * (delta - weights[i]);
                }
                gradScores[i] = sum;
            }
            
            TrainAlignment(t, gradScores, learningRate);
        }

        return gradEncoderHidden;
    }

    public override TTensor[] Forward(TTensor[] inputs, TTensor initialHidden, TTensor initialCell, bool cache = true)
    {
        var encoderOutputs = Encoder(inputs, cache);
        var context = Attention(encoderOutputs, initialHidden);

        var modifiedEncoderOutputs = encoderOutputs
            .Select(output => TensorOperations.Concat(output, context))
            .ToArray();

        var decoderOutputs = Decoder(modifiedEncoderOutputs, initialHidden, initialCell, cache);
        return decoderOutputs;
    }

    public override TTensor[] Backward(TTensor[] gradOutputs, double learningRate)
    {
        var dDecoderInputs = DecoderBackward(gradOutputs, learningRate);
        var encoderOutputs = EncoderNewHiddenCache.ToArray();
        var dEncoderOutputs = AttentionBackwards(dDecoderInputs, encoderOutputs, learningRate);
        var dInputs = EncoderBackward(dEncoderOutputs, learningRate);
        AttentionScores.Clear();
        return dInputs;
    }
}

public interface ILstmAttentionTensorOperations<TWeights, TBiases, TTensor> : ILstmTensorOperations<TWeights, TBiases, TTensor>
    where TWeights : notnull
    where TBiases : notnull
    where TTensor : notnull
{
    public TTensor WeightedSum(TTensor[] tensors, double[] weights);
    public TTensor Multiply(TTensor a, double b);
    public double Dot(TTensor a, TTensor b);
}