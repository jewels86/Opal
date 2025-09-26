using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Lstm;

public abstract class LstmAttentionLayer<TWeights, TBiases, TTensor> : LstmLayer<TWeights, TBiases, TTensor>
    where TWeights : notnull
    where TBiases : notnull
    where TTensor : notnull
{
    protected List<double[]> attentionScores = [];
    protected List<TTensor> attentionContexts = [];
    protected new readonly ILstmAttentionTensorOperations<TWeights, TBiases, TTensor> tensorOperations;
    
    public LstmAttentionLayer(int[] inputShape, int[] hiddenShape, int[] outputShape,
        ILstmAttentionTensorOperations<TWeights, TBiases, TTensor> tensorOperations, IOptimizer<TWeights, TBiases> optimizer,
        ActivationFunction<TTensor> sigmoidActivation, ActivationFunction<TTensor> tanhActivation)
        : base(inputShape, hiddenShape, outputShape, tensorOperations, optimizer, sigmoidActivation, tanhActivation)
    {
        this.tensorOperations = tensorOperations;
    }

    public abstract double[] Alignment(TTensor[] hidden, TTensor prevState);
    public abstract void TrainAlignment(int timeStep, double[] gradScores, double learningRate);

    public TTensor Attention(TTensor[] hidden, TTensor prevState)
    {
        double[] scores = Alignment(hidden, prevState);
        double[] weights = Tensors.Softmax(scores);
        TTensor context = tensorOperations.WeightedSum(hidden, weights);
        attentionScores.Add(scores);
        attentionContexts.Add(context);
        return context;
    }

    public TTensor[] AttentionBackwards(TTensor[] gradOutputs, TTensor[] encoderHiddenStates, double learningRate)
    {
        int timeSteps = gradOutputs.Length;
        int encoderSteps = encoderHiddenStates.Length;
        TTensor[] gradEncoderHidden = new TTensor[encoderSteps];
        
        for (int i = 0; i < encoderSteps; i++)
            gradEncoderHidden[i] = tensorOperations.DefaultState(HiddenShape);

        for (int t = 0; t < timeSteps; t++)
        {
            double[] scores = attentionScores[t];
            double[] weights = Tensors.Softmax(scores);
            
            TTensor gradContext = gradOutputs[t];

            for (int i = 0; i < encoderSteps; i++)
            {
                TTensor encoderHidden = encoderHiddenStates[i];
                gradEncoderHidden[i] = tensorOperations.Add(
                    gradEncoderHidden[i],
                    tensorOperations.Multiply(gradContext, weights[i])
                );
            }
            
            double[] gradWeights = new double[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                TTensor encoderHidden = encoderHiddenStates[i];
                gradWeights[i] = tensorOperations.Dot(gradContext, encoderHidden);
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
            .Select(output => tensorOperations.Concat(output, context))
            .ToArray();

        var decoderOutputs = Decoder(modifiedEncoderOutputs, initialHidden, initialCell, cache);
        return decoderOutputs;
    }

    public override TTensor[] Backward(TTensor[] gradOutputs, double learningRate)
    {
        var dDecoderInputs = DecoderBackward(gradOutputs, learningRate);
        var encoderOutputs = encoderNewHiddenCache.ToArray();
        var dEncoderOutputs = AttentionBackwards(dDecoderInputs, encoderOutputs, learningRate);
        var dInputs = EncoderBackward(dEncoderOutputs, learningRate);
        attentionScores.Clear();
        attentionContexts.Clear();
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