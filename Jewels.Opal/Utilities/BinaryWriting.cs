namespace Jewels.Opal.Utilities;

public static class BinaryWriting
{
    public static void WriteMatrix(BinaryWriter writer, float[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        writer.Write(rows);
        writer.Write(cols);
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) 
            writer.Write(matrix[i, j]);
    }
    public static float[,] ReadMatrix(BinaryReader reader)
    {
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        var matrix = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) 
            matrix[i, j] = (float)reader.ReadDouble();
        return matrix;
    }
    public static void WriteVector(BinaryWriter writer, float[] vector)
    {
        writer.Write(vector.Length);
        foreach (float t in vector) writer.Write(t);
    }
    public static float[] ReadVector(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        var v = new float[len];
        for (int i = 0; i < len; i++) v[i] = (float)reader.ReadDouble();
        return v;
    }
    
    public static void WriteShape(BinaryWriter writer, int[] shape)
    {
        writer.Write(shape.Length);
        foreach (int t in shape) writer.Write(t);
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