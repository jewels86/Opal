namespace Opal.Utilities;

public class Tensor
{
    public readonly double[] Data;
    public int[] Shape { get; }

    public Tensor(params int[] shape)
    {
        Shape = shape;
        Data = new double[Shape.Aggregate(1, (a, b) => a * b)];
    }
    public Tensor(double[] data, int[] shape)
    {
        if (data.Length != shape.Aggregate(1, (a, b) => a * b))
            throw new ArgumentException("Data length does not match shape.");
        Data = data;
        Shape = shape;
    }

    public double this[params int[] indices]
    {
        get => Data[GetFlatIndex(indices)];
        set => Data[GetFlatIndex(indices)] = value;
    }
    
    public Tensor Slice(int axis, int index)
    {
        if (axis < 0 || axis >= Shape.Length)
            throw new ArgumentOutOfRangeException(nameof(axis));
        if (index < 0 || index >= Shape[axis])
            throw new ArgumentOutOfRangeException(nameof(index));
    
        int[] newShape = Shape.Where((_, i) => i != axis).ToArray();
        Tensor result = new Tensor(newShape);
    
        int[] indices = new int[Shape.Length];
        int[] resultIndices = new int[newShape.Length];
    
        void Copy(int dim)
        {
            if (dim == Shape.Length)
            {
                indices[axis] = index;
                result[resultIndices] = this[indices];
                return;
            }
            if (dim == axis)
            {
                Copy(dim + 1);
            }
            else
            {
                int resDim = dim < axis ? dim : dim - 1;
                for (int i = 0; i < Shape[dim]; i++)
                {
                    indices[dim] = i;
                    resultIndices[resDim] = i;
                    Copy(dim + 1);
                }
            }
        }
    
        Copy(0);
        return result;
    }

    private int GetFlatIndex(int[] indices)
    {
        int flatIndex = 0, stride = 1;
        for (int i = Shape.Length - 1; i >= 0; i--)
        {
            flatIndex += indices[i] * stride;
            stride *= Shape[i];
        }
        return flatIndex;
    }
}