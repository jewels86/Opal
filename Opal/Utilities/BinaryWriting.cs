namespace Opal.Utilities;

public static class BinaryWriting
{
    public static void WriteMatrix(BinaryWriter writer, double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        writer.Write(rows);
        writer.Write(cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) writer.Write(matrix[i, j]);
        }
    }
    public static double[,] ReadMatrix(BinaryReader reader)
    {
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        var matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) matrix[i, j] = reader.ReadDouble();
        }
        
        return matrix;
    }
    public static void WriteVector(BinaryWriter writer, double[] vector)
    {
        writer.Write(vector.Length);
        for (int i = 0; i < vector.Length; i++) writer.Write(vector[i]);
    }
    public static double[] ReadVector(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        var v = new double[len];
        for (int i = 0; i < len; i++) v[i] = reader.ReadDouble();
        return v;
    }
    
    public static void WriteShape(BinaryWriter writer, int[] shape)
    {
        writer.Write(shape.Length);
        for (int i = 0; i < shape.Length; i++) writer.Write(shape[i]);
    }

    public static int[] ReadShape(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        var shape = new int[len];
        for (int i = 0; i < len; i++) shape[i] = reader.ReadInt32();
        return shape;
    }
    
    public static void WriteString(BinaryWriter writer, string str)
    {
        writer.Write(str.Length);
        writer.Write(str.ToCharArray());
    }

    public static string ReadString(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        char[] chars = reader.ReadChars(len);
        return new string(chars);
    }
}