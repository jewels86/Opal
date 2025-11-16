using Opal.NNs.Ff;
using Opal.NNs.Recurrent;

namespace Opal.Autograd.Catalogs;

public class ScalarCatalog
{

    public ScalarTensor Add(ScalarTensor a, ScalarTensor b) => a + b;

    public ScalarTensor Subtract(ScalarTensor a, ScalarTensor b) => a - b;
    
    public ScalarTensor Scale(ScalarTensor a, ScalarTensor scale) => a * scale;
    
    public double ZeroGradient(double a) => 0.0;

    public void WriteWeight(BinaryWriter writer, double weight) => writer.Write(weight);

    public double ReadWeight(BinaryReader reader) => reader.ReadDouble();

    public void WriteBias(BinaryWriter writer, double bias) => writer.Write(bias);

    public double ReadBias(BinaryReader reader) => reader.ReadDouble();
    
    public void WriteState(BinaryWriter writer, double state) => writer.Write(state);
    public double ReadState(BinaryReader reader) => reader.ReadDouble();
}